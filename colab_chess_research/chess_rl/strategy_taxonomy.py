"""Stable strategy tags. Tags describe a corpus, not a claim of an optimal move."""

TAXONOMY = {
    "openings": (
        "ruy_lopez",
        "sicilian_defence",
        "nimzo_indian_defence",
        "flank_openings",
        "other",
    ),
    "tactics": (
        "captures",
        "checks",
        "mates",
        "forks",
        "pins",
        "skewers",
        "discovered_attacks",
        "hanging_pieces",
        "sacrifices",
        "queen_sacrifices",
    ),
    "calculation": (
        "forcing_sequences",
        "checks_captures_threats",
        "short_lines",
        "best_continuations",
    ),
    "king_safety": (
        "castling_status",
        "pawn_shield",
        "open_files",
        "attacker_count",
        "exposed_king",
    ),
    "pawn_structure": ("isolated", "doubled", "passed", "backward", "chains", "majorities"),
    "piece_activity": (
        "mobility",
        "outposts",
        "bishop_pair",
        "trapped_pieces",
        "bad_bishops",
        "active_knights",
    ),
    "center_control": ("central_pawns", "attacked_squares", "piece_pressure"),
    "rook_activity": (
        "open_files",
        "semi_open_files",
        "seventh_rank",
        "behind_passed_pawns",
        "rook_endgames",
    ),
    "endgames": (
        "king_activity",
        "promotion_races",
        "rook",
        "minor_piece",
        "queen",
        "conversion",
        "holding",
    ),
    "draw_awareness": (
        "stalemate",
        "insufficient_material",
        "repetition",
        "fifty_move_rule",
        "fortress_like",
    ),
    "annotated_benchmark_games": ("user_provided", "rubinstein"),
}


def validate_theme(theme, subtheme=None):
    if theme not in TAXONOMY:
        raise ValueError(f"Unknown strategy theme: {theme!r}")
    if subtheme not in (None, "", "general") and subtheme not in TAXONOMY[theme]:
        raise ValueError(f"Unknown subtheme {subtheme!r} for {theme}")
    return theme, subtheme or "general"
