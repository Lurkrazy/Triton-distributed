import re
from dataclasses import dataclass
from typing import List, Tuple
import torch
from triton.distributed.kernels.nvidia.common_ops import wait_eq
from triton.distributed.utils import CUDA_CHECK
from cuda import cuda


@dataclass
class AGGEMMManualContext:
    """Lightweight context for manual scheduling."""

    rank: int
    num_ranks: int
    workspace_tensors: List[torch.Tensor]
    barrier_tensors: List[torch.Tensor]
    comm_buf: torch.Tensor
    ag_stream: torch.cuda.Stream
    gemm_stream: torch.cuda.Stream


# ---------------------------------------------------------------------------
# DSL utilities
# ---------------------------------------------------------------------------

def parse_schedule(schedule: str) -> List[Tuple[str, int]]:
    """Parse a schedule string like 'comm0 compute0 comm1 compute1'."""
    tokens = re.findall(r"(comm|compute)(\d+)", schedule)
    return [(t[0], int(t[1])) for t in tokens]


# ---------------------------------------------------------------------------
# Tile-level primitives
# ---------------------------------------------------------------------------

def all_gather_tile(ctx: AGGEMMManualContext, local_tensor: torch.Tensor, tile_idx: int):
    """Copy one tile from peer ``tile_idx`` and set its ready flag."""
    rank = ctx.rank
    src_rank = tile_idx % ctx.num_ranks
    M_per_rank, N = local_tensor.shape

    stream = ctx.ag_stream
    with torch.cuda.stream(stream):
        if src_rank == rank:
            return
        dst = ctx.workspace_tensors[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
        src = ctx.workspace_tensors[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
        dst.copy_(src)
        err, = cuda.cuStreamWriteValue32(
            stream.cuda_stream,
            ctx.barrier_tensors[rank][src_rank].data_ptr(),
            1,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)


def gemm_tile(ctx: AGGEMMManualContext, B: torch.Tensor, C: torch.Tensor, tile_idx: int):
    """Compute ``C_i = A_i @ B.T`` for tile ``tile_idx``."""
    rank = ctx.rank
    M_per_rank = C.shape[0] // ctx.num_ranks
    slice_rows = slice(tile_idx * M_per_rank, (tile_idx + 1) * M_per_rank)

    wait_eq(
        ctx.barrier_tensors[rank][tile_idx].data_ptr(),
        1,
        ctx.gemm_stream,
    )

    with torch.cuda.stream(ctx.gemm_stream):
        A_tile = ctx.workspace_tensors[rank][slice_rows, :]
        C[slice_rows] = torch.matmul(A_tile, B.t())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_manual_schedule(schedule: str, a: torch.Tensor, b: torch.Tensor, ctx: AGGEMMManualContext) -> torch.Tensor:
    """Execute the given schedule string and return the result C."""
    ops = parse_schedule(schedule)
    M_per_rank, K = a.shape
    N_per_rank = b.shape[0]
    C = torch.empty((ctx.num_ranks * M_per_rank, N_per_rank), dtype=a.dtype, device=a.device)

    for op, idx in ops:
        if op == "comm":
            all_gather_tile(ctx, a, idx)
        elif op == "compute":
            gemm_tile(ctx, b, C, idx)
        else:
            raise ValueError(f"Unknown op {op}")

    # ensure streams finish
    torch.cuda.current_stream().wait_stream(ctx.ag_stream)
    torch.cuda.current_stream().wait_stream(ctx.gemm_stream)

    return C

