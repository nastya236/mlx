def scale_tiled_offset(scale_index, num_rows, num_scale_cols):
    """
    Compute the tiled layout offset for scale factors used in tensor cores.

    Args:
        scale_index: Linear scale index in the original layout
        num_rows: Number of rows in the matrix (M)
        num_scale_cols: Number of scale columns (K / group_size)

    Returns:
        Offset in the tiled layout
    """
    # Convert linear scale index to 2D coordinates
    row = scale_index // num_scale_cols
    col = scale_index % num_scale_cols

    rows_per_tile = 128
    rows_per_sub_block = 32
    cols_per_sub_block = 4
    sub_blocks_per_tile = 4  # Vertically stacked

    # Decompose row position
    tile_row = row // rows_per_tile  # Which tile row
    row_in_tile = row % rows_per_tile  # Row within tile
    sub_block_row = row_in_tile // rows_per_sub_block  # Sub-block within tile (0-3)
    row_in_sub_block = row_in_tile % rows_per_sub_block  # Row in sub-block (0-31)

    # Decompose column position
    col_tile = col // cols_per_sub_block  # Which column tile
    col_in_sub_block = col % cols_per_sub_block  # Column within sub-block (0-3)

    # Compute tile index
    num_col_tiles = (
        num_scale_cols + cols_per_sub_block - 1
    ) // cols_per_sub_block  # ceil_div
    tile_idx = tile_row * num_col_tiles + col_tile

    # Offset within tile
    # Memory layout: [tile][row_in_sub_block][sub_block_row][col_in_sub_block]
    # Dimensions: [tile][32][4][4], strides: [512][16][4][1]
    offset_in_tile = (
        (row_in_sub_block * sub_blocks_per_tile * cols_per_sub_block)
        + (sub_block_row * cols_per_sub_block)
        + col_in_sub_block
    )

    tile_size = rows_per_tile * cols_per_sub_block  # 512
    return tile_idx * tile_size + offset_in_tile


# Test function
def test_scale_tiled_offset():
    """Test with examples"""

    # Example 1: First tile, first element
    M, K, group_size = 128, 64, 16
    num_scale_cols = K // group_size  # 4 columns

    print("Test 1: Single tile (128 x 4)")
    print(f"Matrix: M={M}, K={K}, group_size={group_size}")
    print(f"Scales shape: ({M}, {num_scale_cols})")
    print()

    # Test corner cases
    test_cases = [
        (0, 0, "First element (row=0, col=0)"),
        (1, 0, "Second element (row=0, col=1)"),
        (4, 0, "Fifth element (row=1, col=0)"),
        (32, 0, "First element of 2nd sub-block (row=32, col=0)"),
        (127, 3, "Last element of tile (row=127, col=3)"),
    ]

    for linear_idx, expected_col, desc in test_cases:
        row = linear_idx // num_scale_cols
        col = linear_idx % num_scale_cols
        tiled_idx = scale_tiled_offset(linear_idx, M, num_scale_cols)
        print(f"{desc}")
        print(f"  Linear index: {linear_idx} -> (row={row}, col={col})")
        print(f"  Tiled offset: {tiled_idx}")
        print()

    # Example 2: Multiple tiles
    print("\nTest 2: Multiple tiles (256 x 8)")
    M, K, group_size = 256, 128, 16
    num_scale_cols = K // group_size  # 8 columns
    print(f"Matrix: M={M}, K={K}, group_size={group_size}")
    print(f"Scales shape: ({M}, {num_scale_cols})")
    print(f"Number of tiles: {(M // 128) * ((num_scale_cols + 3) // 4)}")
    print()

    # First element of each tile
    test_cases = [
        (0, "Tile 0 start (row=0, col=0)"),
        (4, "Tile 1 start (row=0, col=4)"),
        (8 * 128, "Tile 2 start (row=128, col=0)"),
        (8 * 128 + 4, "Tile 3 start (row=128, col=4)"),
    ]

    for linear_idx, desc in test_cases:
        row = linear_idx // num_scale_cols
        col = linear_idx % num_scale_cols
        tiled_idx = scale_tiled_offset(linear_idx, M, num_scale_cols)
        print(f"{desc}")
        print(f"  Linear index: {linear_idx} -> (row={row}, col={col})")
        print(f"  Tiled offset: {tiled_idx}")
        print()


if __name__ == "__main__":
    test_scale_tiled_offset()

    # Verify against NVIDIA's formula
    print("\n" + "=" * 60)
    print("Verification against NVIDIA formula:")
    print("=" * 60)

    def scale_factor_swizzled_offset_nvidia(row_idx, col_idx, col_length):
        """NVIDIA's implementation from example.cu"""
        kTotalRowsPerBaseBlock = 128
        kRowsPerBaseBlockCol = 32
        kColsPerBaseBlockCol = 4

        rb = row_idx // kTotalRowsPerBaseBlock
        rem = row_idx % kTotalRowsPerBaseBlock
        d4 = rem // kRowsPerBaseBlockCol
        d3 = rem % kRowsPerBaseBlockCol
        cbg = col_idx // kColsPerBaseBlockCol
        d5 = col_idx % kColsPerBaseBlockCol

        cbg_cnt = (col_length + kColsPerBaseBlockCol - 1) // kColsPerBaseBlockCol
        return (
            ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16
            + d4 * kColsPerBaseBlockCol
            + d5
        )

    # Compare both implementations
    M, K, group_size = 256, 128, 16
    num_scale_cols = K // group_size

    print(f"Testing {M * num_scale_cols} scales...")
    all_match = True
    mismatches = []

    for scale_idx in range(M * num_scale_cols):
        row = scale_idx // num_scale_cols
        col = scale_idx % num_scale_cols

        our_offset = scale_tiled_offset(scale_idx, M, num_scale_cols)
        nvidia_offset = scale_factor_swizzled_offset_nvidia(row, col, num_scale_cols)

        if our_offset != nvidia_offset:
            all_match = False
            mismatches.append((scale_idx, our_offset, nvidia_offset))

    if all_match:
        print("✅ ALL OFFSETS MATCH! Implementation is correct.")
    else:
        print(f"❌ MISMATCH! {len(mismatches)} differences found.")
        print("\nFirst 10 mismatches:")
        for i, (idx, ours, nvidia) in enumerate(mismatches[:10]):
            row = idx // num_scale_cols
            col = idx % num_scale_cols
            print(f"  Index {idx} (row={row}, col={col}): ours={ours}, nvidia={nvidia}")
