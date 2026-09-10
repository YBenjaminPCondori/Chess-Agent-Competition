# Encoding Specification

## Board Version board_v1

The tensor is float32 with shape [21,8,8]. Rank index 0 is chess rank 1; file index 0 is file a. Black positions are not rotated.

| Planes | Contents |
| --- | --- |
| 0-5 | White P, N, B, R, Q, K |
| 6-11 | Black P, N, B, R, Q, K |
| 12 | White-to-move constant: 1 for White, 0 for Black |
| 13-16 | White kingside, White queenside, Black kingside, Black queenside rights |
| 17 | Legal en-passant target square, or zero |
| 18 | min(halfmove_clock,100)/100 |
| 19 | min(board.ply(),600)/600 |
| 20 | Always zero; repetition history is unavailable in standalone FEN |

Rights are distinct from immediate castling legality. Feature normalization is identical in data preparation, self-play, model inference, and export.

## Action Version action_v1

The policy output is contiguous [73,8,8], flattened to 4,672 values. Index = move_plane * 64 + from_square, where from_square = rank * 8 + file.

Planes 0-55 use directions N, NE, E, SE, S, SW, W, NW, each followed by distances 1 through 7. Planes 56-63 use (file,rank) offsets (1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2).

Planes 64-72 use promotion pieces knight, bishop, rook. For each piece, absolute file offsets are -1, 0, +1. Rank displacement is +1 for White and -1 for Black. Queen promotions use ordinary displacement planes and are reconstructed as queen promotions when a pawn reaches its last rank.

Castling uses the legal king's two-square displacement. En passant uses the pawn displacement. The legal mask is constructed by encoding python-chess legal moves, not by trusting geometry alone. Decoding rejects illegal/off-board actions. Terminal positions have no policy target.

Legal-only cross-entropy smooths only over legal actions. Illegal positions in the logits have zero gradient from policy loss. The tests exercise complete legal-move round trips on both fixed edge cases and seeded legal games.
