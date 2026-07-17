import os
import re
from typing import Dict, List, Tuple

import pandas as pd


from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
DATA_PATHS = [
    os.path.join(ROOT_DIR, "steam_clean.csv.gz"),
    os.path.join(ROOT_DIR, "Data_sets", "steam_clean.csv.gz"),
]

# ── Exact 15 games the user wants, in rank order ─────────────────────────────
GAME_OPTIONS_ORDERED = [
    "PLAYERUNKNOWN'S BATTLEGROUNDS",
    "Dota 2",
    "Counter-Strike: Global Offensive",
    "Tom Clancy's Rainbow Six Siege",
    "Warframe",
    "Grand Theft Auto V",
    "Team Fortress 2",
    "MONSTER HUNTER: WORLD",
    "ARK: Survival Evolved",
    "Rust",
    "Rocket League",
    "Garry's Mod",
    "Path of Exile",
    "Football Manager 2018",
    "Sid Meier's Civilization V",
]

# ── Display name → folder name (covers both apostrophe variants) ──────────────
GAME_DISPLAY_TO_FOLDER = {
    # PUBG — two folder variants exist (with/without apostrophe)
    "PLAYERUNKNOWN'S BATTLEGROUNDS": "PLAYERUNKNOWN'S_BATTLEGROUNDS",
    # Dota 2
    "Dota 2": "Dota_2",
    # CS:GO
    "Counter-Strike: Global Offensive": "Counter-Strike_Global_Offensive",
    "Counter-Strike Global Offensive": "Counter-Strike_Global_Offensive",
    # R6
    "Tom Clancy's Rainbow Six Siege": "Tom_Clancy's_Rainbow_Six_Siege",
    # Warframe
    "Warframe": "Warframe",
    # GTA V
    "Grand Theft Auto V": "Grand_Theft_Auto_V",
    # TF2
    "Team Fortress 2": "Team_Fortress_2",
    # MHW
    "MONSTER HUNTER: WORLD": "MONSTER_HUNTER_WORLD",
    "Monster Hunter World": "MONSTER_HUNTER_WORLD",
    # ARK
    "ARK: Survival Evolved": "ARK_Survival_Evolved",
    "ARK Survival Evolved": "ARK_Survival_Evolved",
    # Rust
    "Rust": "Rust",
    # Rocket League
    "Rocket League": "Rocket_League",
    # Garry's Mod — two folder variants
    "Garry's Mod": "Garry's_Mod",
    # Path of Exile
    "Path of Exile": "Path_of_Exile",
    # Football Manager 2018
    "Football Manager 2018": "Football_Manager_2018",
    # Civ V — two folder variants
    "Sid Meier's Civilization V": "Sid_Meier's_Civilization_V",
}

# ── Alias → canonical display name ────────────────────────────────────────────
GAME_ALIASES = {
    # PUBG
    "PLAYERUNKNOWN'S BATTLEGROUNDS": "PLAYERUNKNOWN'S BATTLEGROUNDS",
    "PLAYERUNKNOWNS BATTLEGROUNDS": "PLAYERUNKNOWN'S BATTLEGROUNDS",
    "PUBG": "PLAYERUNKNOWN'S BATTLEGROUNDS",
    # Dota
    "Dota 2": "Dota 2",
    "DotA 2": "Dota 2",
    # CS:GO
    "Counter-Strike: Global Offensive": "Counter-Strike: Global Offensive",
    "Counter-Strike Global Offensive": "Counter-Strike: Global Offensive",
    "Counter Strike Global Offensive": "Counter-Strike: Global Offensive",
    # R6
    "Tom Clancy's Rainbow Six Siege": "Tom Clancy's Rainbow Six Siege",
    "Rainbow Six Siege": "Tom Clancy's Rainbow Six Siege",
    # Warframe
    "Warframe": "Warframe",
    # GTA V
    "Grand Theft Auto V": "Grand Theft Auto V",
    "GTA V": "Grand Theft Auto V",
    # TF2
    "Team Fortress 2": "Team Fortress 2",
    # MHW
    "MONSTER HUNTER: WORLD": "MONSTER HUNTER: WORLD",
    "Monster Hunter World": "MONSTER HUNTER: WORLD",
    "MONSTER HUNTER WORLD": "MONSTER HUNTER: WORLD",
    # ARK
    "ARK: Survival Evolved": "ARK: Survival Evolved",
    "ARK Survival Evolved": "ARK: Survival Evolved",
    # Rust
    "Rust": "Rust",
    # Rocket League
    "Rocket League": "Rocket League",
    # Garry's Mod
    "Garry's Mod": "Garry's Mod",
    "Garrys Mod": "Garry's Mod",
    # Path of Exile
    "Path of Exile": "Path of Exile",
    # Football Manager 2018
    "Football Manager 2018": "Football Manager 2018",
    # Civ V
    "Sid Meier's Civilization V": "Sid Meier's Civilization V",
    "Sid Meiers Civilization V": "Sid Meier's Civilization V",
}

HORIZON_LABELS = {
    "Next 1 Hour": 1,
    "Next 3 Hours": 3,
    "Next 6 Hours": 6,
    "Next 12 Hours": 12,
    "Next 24 Hours": 24,
    "Next 48 Hours": 48,
}
HORIZON_HOURS = {k: v for k, v in HORIZON_LABELS.items()}


def _resolve_data_path() -> str:
    for path in DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Unable to locate the Steam dataset.")


def load_steam_data() -> pd.DataFrame:
    data_path = _resolve_data_path()
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    df = df.sort_values(["title", "datetime"]).reset_index(drop=True)
    if "current_players" in df.columns:
        df["current_players"] = pd.to_numeric(df["current_players"], errors="coerce").fillna(0)
    return df


def _normalize_game_name(game_name: str, available_titles: List[str]) -> str:
    if not game_name:
        raise ValueError("Game name is empty")

    normalized = str(game_name).strip()
    if normalized in available_titles:
        return normalized

    # Check alias table
    alias = GAME_ALIASES.get(normalized)
    if alias and alias in available_titles:
        return alias

    # Fuzzy: strip non-alphanumeric for comparison
    compact = re.sub(r"[^a-z0-9]+", "", normalized.lower())
    for title in available_titles:
        title_compact = re.sub(r"[^a-z0-9]+", "", str(title).lower())
        if compact == title_compact:
            return title

    # Partial match
    for title in available_titles:
        if normalized.lower() in str(title).lower() or str(title).lower() in normalized.lower():
            return title

    raise ValueError(f"Game '{game_name}' not found in datasource")


def prepare_game_history(data: pd.DataFrame, game_name: str) -> pd.Series:
    available_titles = [str(title).strip() for title in data["title"].dropna().unique().tolist()]
    resolved_game = _normalize_game_name(game_name, available_titles)
    game_data = data.loc[data["title"] == resolved_game].copy()
    if game_data.empty:
        raise ValueError("Game not found in datasource")
    game_data = game_data.sort_values("datetime")
    history = pd.to_numeric(game_data["current_players"], errors="coerce").fillna(0)
    history.index = pd.DatetimeIndex(game_data["datetime"])
    return history.astype(float)


def get_game_options(data: pd.DataFrame) -> List[str]:
    """Return exactly the 15 canonical games in rank order."""
    return list(GAME_OPTIONS_ORDERED)
