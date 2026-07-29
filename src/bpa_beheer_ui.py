"""
BPA Jaarlijks Beheer Tool – Streamlit UI
=========================================
Start met:
    streamlit run src/bpa_beheer_ui.py

Vereist:
    pip install streamlit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import json

# Hergebruik alle logica uit bpa_beheer.py
from bpa_beheer import (
    laad_config,
    sla_config_op,
    bereken_overzicht,
    bouw_model_kosten,
    laad_excel_onderdelen,
    laad_classificatie_selectie,
    regionale_adoptie_parameter,
    adoptie_kans,
    aantal_klanten_per_component,
    binomiale_verdeling,
    binomiale_quantile,
    verwacht_subscripties_per_component,
    gevoeligheid_verwachte_z,
    pareto_alpha_X,
    optimale_alpha_bij_X,
    beta_r_winstband,
    greedy_alpha_sweep,
    winst_voor_wtp_grid,
    metrieken_voor_wtp_grid,
    greedy_detail_for_params,
    SERVICE_LEVELS,
    CONFIG_PATH,
    HISTORY_PATH,
    SCRIPT_DIR,
    SELECTIE_PATH,
    EXCEL_PATH,
    SUBSCRIPTIES_PATH,
)
from classificatie import (
    ClassificatieParams,
    voer_classificatie_uit,
    schrijf_selectie_json,
    controleer_kolommen,
    laad_ruwe_dataset,
    bereken_scores,
    pas_basis_filters_toe,
    pas_topn_selectie_toe,
    bouw_selectie_payload,
    weight_sensitivity,
)
from model import BPAOptimizationModel

# ══════════════════════════════════════════════════════════════════════════════
#  CACHE-WRAPPERS  (sterk versnellen Streamlit-reruns)
# ══════════════════════════════════════════════════════════════════════════════
#
# Streamlit voert dit script opnieuw uit bij élke widget-interactie. Zonder
# caching wordt de (grote) Excel telkens opnieuw geparsed en doorloopt
# `bereken_overzicht` weer alle componenten. De wrappers hieronder zorgen dat
# we alleen herrekenen als (a) een bron-bestand op disk gewijzigd is óf
# (b) de gebruiker de config heeft aangepast. Cache wordt automatisch
# ongeldig zodra een van die inputs verandert.

def _file_mtime(path: str) -> float:
    """Return mtime in seconds; 0.0 als bestand ontbreekt."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_laad_classificatie_selectie(_mtime: float) -> dict:
    """Cached versie van laad_classificatie_selectie — keyed op bestand-mtime."""
    return laad_classificatie_selectie()


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_laad_ruwe_dataset(_excel_mtime: float, sheet_name, upload=None) -> pd.DataFrame:
    """Cache de (trage) Excel-parse voor de classificatie.

    Keyed op bestand-mtime + sheet voor de repo-Excel, of op de geüploade
    file-inhoud (Streamlit hasht een UploadedFile op inhoud, dus de parameter
    krijgt GEEN underscore-prefix — anders zou een tweede upload met dezelfde
    sheet-naam onterecht de vorige cache-hit teruggeven). Hierdoor wordt de
    Excel maar één keer geparsed per uniek bestand; daarna gaan parameter-tweaks
    razendsnel omdat alleen de gevectoriseerde scoring opnieuw draait.
    """
    if upload is not None:
        # Reset de leespositie: een eerder gelezen/gehashte buffer kan aan het
        # einde staan, waardoor pd.read_excel niets zou inlezen.
        try:
            upload.seek(0)
        except (AttributeError, ValueError):
            pass
        bron = upload
    else:
        bron = EXCEL_PATH
    return laad_ruwe_dataset(bron, sheet_name=sheet_name)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_bereken_overzicht(cfg_json: str, _excel_mtime: float, _selectie_mtime: float) -> pd.DataFrame:
    """Cached versie van bereken_overzicht — keyed op JSON-config + bestand-mtimes."""
    return bereken_overzicht(json.loads(cfg_json))


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_aantal_klanten(_excel_mtime: float, upload=None) -> pd.Series:
    """Cached M_i per component. upload = bestandspad (str) of UploadedFile."""
    return aantal_klanten_per_component(upload)


def get_classificatie_info() -> dict:
    """Lees bpa_selectie.json (cached). Auto-invalideert bij file-update."""
    return _cached_laad_classificatie_selectie(_file_mtime(SELECTIE_PATH))


def get_overzicht_df(cfg: dict) -> pd.DataFrame:
    """Bereken het overzicht (cached). Auto-invalideert bij config- of bestand-wijziging."""
    cfg_json = json.dumps(cfg, sort_keys=True, default=str)
    return _cached_bereken_overzicht(
        cfg_json,
        _file_mtime(EXCEL_PATH),
        _file_mtime(SELECTIE_PATH),
    )


def invalidate_caches() -> None:
    """Forceer een verse Excel/JSON-read bij volgende aanroep."""
    _cached_bereken_overzicht.clear()
    _cached_laad_classificatie_selectie.clear()
    _cached_laad_ruwe_dataset.clear()
    _cached_aantal_klanten.clear()


@st.cache_data(show_spinner="Gewichten-sweep berekenen…", max_entries=8)
def _cached_weight_sweep(_df_scored: pd.DataFrame, params_json: str, step: float,
                         versie: int = 2):
    """Cached gewicht-sweep. `_df_scored` (leidende underscore) wordt NIET
    gehasht; de cache-sleutel is `params_json` + `step` + `versie`. De
    versie-token wordt opgehoogd wanneer de output-vorm wijzigt (bv. nieuwe
    rangorde-kolommen), zodat oude cache-resultaten automatisch verlopen.
    """
    p = json.loads(params_json)
    params = ClassificatieParams(
        threshold=p["threshold"],
        selectie_modus=p["selectie_modus"],
        top_n=p["top_n"],
        weight_prijs=p["weight_prijs"],
        weight_locaties=p["weight_locaties"],
        weight_orders=p["weight_orders"],
        orders_power=p["orders_power"],
        min_prijs=p.get("min_prijs", 0.0),
        min_orders=p.get("min_orders", 0.0),
        min_klantlocaties=p["min_klantlocaties"],
        article_type_filter=tuple(p["article_type_filter"]),
        score_methode=p.get("score_methode", "arithmetisch"),
        epsilon=p.get("epsilon", 1.0),
    )
    return weight_sensitivity(_df_scored, params, step=step, return_combos=True)


def representatieve_z(default: int = 1) -> int:
    """Representatief aantal subscripties (Z) uit het huidige overzicht.

    Vervangt de oude globale standaardwaarde: neemt de mediaan van het
    werkelijke aantal klantlocaties (n_klanten) over alle componenten. Wordt
    gebruikt als referentie-/startwaarde in de gevoeligheidsgrafieken.
    """
    _df = st.session_state.get("overzicht_df")
    if _df is not None and not _df.empty and "n_klanten" in _df.columns:
        _med = pd.to_numeric(_df["n_klanten"], errors="coerce").median()
        if pd.notna(_med) and _med >= 1:
            return int(round(_med))
    return default


# ══════════════════════════════════════════════════════════════════════════════
#  PAGINA-INSTELLINGEN
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BPA Beheer Tool",
    page_icon="⚙️",
    layout="wide",
)

_logo_path = os.path.join(os.path.dirname(__file__), "BPA.png")
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists(_logo_path):
        st.image(_logo_path, width=120)
with col_title:
    st.title("BPA Jaarlijks Beheer Tool")
    st.caption(f"Configuratiebestand: `{CONFIG_PATH}`")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG IN SESSION STATE LADEN
# ══════════════════════════════════════════════════════════════════════════════

if "cfg" not in st.session_state:
    st.session_state.cfg = laad_config()

cfg = st.session_state.cfg

# ── Excel altijd uit de repository ────────────────────────────────────────
_excel_file = None  # gebruik altijd EXCEL_PATH uit de repo

# Overzicht altijd vers berekenen bij opstarten van de sessie
if "overzicht_df" not in st.session_state:
    with st.spinner("Excel laden en basisvoorraden berekenen…"):
        _df = get_overzicht_df(cfg)
    if not _df.empty:
        st.session_state.overzicht_df = _df

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_overzicht, tab_subscripties, tab_toevoegen, tab_verwijderen, tab_historie, tab_kosten, tab_drempel, tab_classificatie = st.tabs([
    "📊 Overzicht",
    "✏️ Subscripties aanpassen",
    "➕ Component toevoegen",
    "🗑️ Component verwijderen",
    "📈 Historiek",
    "💰 Kostenanalyse",
    "🔢 Subscriptiedrempel",
    "🏷️ Classificatie",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 – OVERZICHT
# ─────────────────────────────────────────────────────────────────────────────

with tab_overzicht:
    st.subheader("Basisvoorraden per component")

    # Excel-bestandsdatum ophalen
    try:
        from bpa_beheer import EXCEL_PATH as _EXCEL_PATH
        _excel_mtime = date.fromtimestamp(os.path.getmtime(_EXCEL_PATH)).isoformat()
    except Exception:
        _excel_mtime = "onbekend"

    st.write(
        f"Configuratie bijgewerkt: **{cfg['aangepast']}** · "
        f"Excel gewijzigd: **{_excel_mtime}**"
    )

    # ── Classificatie-koppeling status ────────────────────────────────────
    _cls_info = get_classificatie_info()
    if _cls_info:
        _lt_ov = _cls_info.get('lt_overzicht', {})
        _n_cls = len(_cls_info.get('items', {}))
        st.success(
            f"🔗 Classificatie-koppeling actief — **{_n_cls}** componenten geselecteerd "
            f"(gegenereerd {_cls_info.get('gegenereerd', '?')}). "
            f"LT-bron: ✅ geupdate **{_lt_ov.get('geupdate', 0)}**  ·  "
            f"⚠️ ERP-default **{_lt_ov.get('default', 0)}**  ·  "
            f"❌ ontbreekt **{_lt_ov.get('ontbreekt', 0)}**"
        )
    else:
        st.info(
            f"ℹ️  Geen classificatie-selectie gevonden ({SELECTIE_PATH}). "
            f"Draai `classificatie_scoring.py` om de koppeling te activeren."
        )

    if st.button("🔄 Herbereken (laadt Excel opnieuw)"):
        invalidate_caches()
        with st.spinner("Berekenen…"):
            df = get_overzicht_df(cfg)
        if df.empty:
            st.warning("Geen onderdelen gevonden.")
        else:
            st.session_state.overzicht_df = df
            st.rerun()

    if "overzicht_df" in st.session_state:
        df = st.session_state.overzicht_df
        sl_cols = [c for c in df.columns if c.startswith("s@")]

        # Samenvattingsregel
        totals = {c: int(df[c].sum()) for c in sl_cols}
        st.write("**Totale basisvoorraad:**  " +
                 "  |  ".join(f"`{c}` = **{totals[c]}**" for c in sl_cols))

        # Aandeel S* > 1 — extra voorraadkosten bovenop S*=1
        if sl_cols and 'IP' in df.columns:
            _parts = []
            for _sc in sl_cols:
                _ip_vals     = df['IP'].fillna(0)
                _extra_units = (df[_sc] - 1).clip(lower=0)          # max(S*-1, 0) per component
                _base_cost   = _ip_vals.sum()                        # Σ 1 × IP (S*=1 scenario)
                _extra_cost  = (_extra_units * _ip_vals).sum()       # Σ (S*-1) × IP
                _total_cost  = (df[_sc] * _ip_vals).sum()
                _pct_extra   = _extra_cost / _total_cost * 100 if _total_cost > 0 else 0.0
                _n_gt1       = int((df[_sc] > 1).sum())
                _parts.append(
                    f"`{_sc}` → **{_n_gt1}** comp. met S\u002a > 1, "
                    f"extra kost boven S\u002a=1: **€ {_extra_cost:,.0f}** (**{_pct_extra:.1f}%** van totale inv.)"
                )
            if _parts:
                st.caption("Extra inv. bovenop S\u002a=1:  \n" + "  \n".join(_parts))

        # Laad vorige snapshot voor Δ-kolommen
        _prev_comp = {}
        _prev_datum = None
        # 1) Voorkeur: vorige overzicht_df uit session_state (vastgelegd bij opslaan)
        _prev_df = st.session_state.get("overzicht_df_prev")
        if _prev_df is not None and not _prev_df.empty:
            _prev_datum = "vorige opgeslagen staat"
            for _code in _prev_df.index:
                _prev_comp[str(_code)] = {
                    _sc: int(_prev_df.at[_code, _sc])
                    for _sc in _prev_df.columns if _sc.startswith("s@")
                }
        # 2) Fallback: oudere snapshot uit history-bestand (legacy)
        elif os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, encoding='utf-8') as _fh:
                    _hist_ov = json.load(_fh)
                for _snap_ov in reversed(_hist_ov):
                    if 'componenten' in _snap_ov:
                        _prev_comp  = _snap_ov['componenten']
                        _prev_datum = _snap_ov['datum']
                        break
            except Exception:
                pass

        # Bouw weergave-df met Δ-kolommen
        _df_disp = df.reset_index().copy()
        _delta_cols = []
        # Vectoriseer: bouw één lookup-DataFrame van vorige S*-waarden per Code,
        # zodat we per SL-kolom alleen een Series-aftrekking nodig hebben
        # (i.p.v. .apply(axis=1) — orde van grootte sneller bij veel rijen).
        if _prev_comp and sl_cols:
            _prev_df_lookup = (
                pd.DataFrame.from_dict(_prev_comp, orient="index")
                  .reindex(columns=sl_cols)
                  .apply(pd.to_numeric, errors="coerce")
            )
            _codes_str = _df_disp["Code"].astype(str)
            for _sc in sl_cols:
                _dc = f"\u0394{_sc}"
                _delta_cols.append(_dc)
                _prev_series = _codes_str.map(_prev_df_lookup[_sc])
                _df_disp[_dc] = (
                    pd.to_numeric(_df_disp[_sc], errors="coerce") - _prev_series
                )
        else:
            # Geen vorige snapshot beschikbaar — vul Δ-kolommen met NaN
            for _sc in sl_cols:
                _dc = f"\u0394{_sc}"
                _delta_cols.append(_dc)
                _df_disp[_dc] = float("nan")

        def _fmt_delta(v):
            if pd.isna(v): return '\u2014'
            iv = int(v)
            return f"+{iv}" if iv > 0 else str(iv)

        def _style_delta(v):
            if pd.isna(v): return ''
            if v > 0: return 'background-color: #f8d7da'   # rood: omhoog
            if v < 0: return 'background-color: #cce5ff'   # blauw: omlaag
            return 'background-color: #d4edda'              # groen: gelijk

        if _prev_datum:
            st.caption(f"\u0394 ten opzichte van snapshot: **{_prev_datum}**")
        else:
            st.caption("\u0394-kolommen beschikbaar na eerste snapshot (tabblad \U0001f4c8 Historiek).")

        # ── Investering per component ──────────────────────────────────────
        _kp_ov = st.session_state.get('kosten_params', {})
        _sl_ov = _kp_ov.get('service_level', 0.990)
        _sl_ov_col = f"s@{_sl_ov:.1%}"
        # Fallback: gebruik de eerste beschikbare SL-kolom als de gewenste er niet in zit
        if _sl_ov_col not in df.columns and sl_cols:
            _sl_ov_col = sl_cols[0]
            _sl_ov = float(_sl_ov_col[2:-1]) / 100
        if _sl_ov_col in df.columns and 'IP' in df.columns:
            _df_disp['Inv. (€)'] = (_df_disp[_sl_ov_col] * df['IP'].values).round(2)
            _inv_totaal = _df_disp['Inv. (€)'].sum()
            _df_disp['Inv. %'] = (
                (_df_disp['Inv. (€)'] / _inv_totaal * 100).round(1)
                if _inv_totaal > 0 else 0.0
            )
            _inv_cols = ['Inv. (€)', 'Inv. %']
            st.caption(
                f"Inv. (€) = S\u002a × IP bij **{_sl_ov_col}** · "
                f"Totale voorraadwaarde: **€ {_inv_totaal:,.0f}** · "
                f"_(pas service level aan via tabblad 💰 Kostenanalyse)_"
            )
        else:
            _inv_cols = []

        # Tabel
        _fmt_inv  = {c: "{:.0f}" for c in _inv_cols if 'Inv. (€)' in c}
        _fmt_inv |= {c: "{:.1f}%" for c in _inv_cols if 'Inv. %' in c}

        def _style_inv_share(v):
            if pd.isna(v) or _inv_totaal == 0:
                return ''
            intensity = min(int(v / 100 * 255), 255)
            return f'background-color: rgba(25, 118, 210, {v/100:.2f}); color: {"white" if v > 50 else "black"}'

        # ── LT-status kolom (vanuit classificatie-koppeling) ──────────────
        _LT_ICOON = {
            'geupdate':  '✅ geupdate',
            'override':  '✏️ override',
            'default':   '⚠️ ERP-default',
            'ontbreekt': '❌ ontbreekt',
            'handmatig': '🛠 handmatig',
            'onbekend':  '❔ onbekend',
            'nul→30':   '🔵 0→30 dagen',
        }
        if 'LT_bron' in _df_disp.columns:
            _df_disp['LT-status'] = (
                _df_disp['LT_bron'].astype(str).map(_LT_ICOON).fillna(_LT_ICOON['onbekend'])
            )

            def _kleur_lt(v):
                s = str(v)
                if '🔵' in s:             return 'background-color: #bbdefb'  # blauw: LT was 0 → 30
                if '✅' in s or '✏️' in s: return 'background-color: #e8f5e9'
                if '⚠️' in s:              return 'background-color: #fff8e1'
                if '❌' in s:              return 'background-color: #ffebee'
                if '🛠' in s:              return 'background-color: #e3f2fd'
                return ''

            _n_bevest = _df_disp['LT_bron'].isin(['geupdate', 'override', 'handmatig', 'nul→30']).sum()
            _n_warn   = len(_df_disp) - _n_bevest
            if _n_warn > 0:
                st.warning(
                    f"⚠️ {_n_warn}/{len(_df_disp)} componenten hebben een niet-bevestigde "
                    f"levertijd (ERP-default of ontbrekend). Corrigeer via tab "
                    f"**✏️ Subscripties aanpassen** — een ingevulde LT-override telt als bevestigd."
                )

        styled = (
            _df_disp.style
                .format({
                    "lambda_jr": "{:.4f}",
                    "mu":        "{:.4f}",
                    **{c: "{:.0f}" for c in sl_cols},
                    **{dc: _fmt_delta for dc in _delta_cols},
                    **({'Inv. (€)': '€ {:,.0f}', 'Inv. %': '{:.1f}%'} if _inv_cols else {}),
                })
                .map(_style_delta, subset=_delta_cols)
        )
        if 'LT-status' in _df_disp.columns:
            styled = styled.map(_kleur_lt, subset=['LT-status'])
        if _inv_cols and 'Inv. %' in _df_disp.columns:
            styled = styled.map(_style_inv_share, subset=['Inv. %'])

        st.dataframe(styled, use_container_width=True, height=500)

        # Download
        csv = df.to_csv(sep=";", decimal=",").encode("utf-8")
        st.download_button(
            label="⬇️ Download als CSV",
            data=csv,
            file_name=f"bpa_base_stock_{date.today()}.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 – SUBSCRIPTIES / IP / LEVERTIJD AANPASSEN
# ─────────────────────────────────────────────────────────────────────────────

with tab_subscripties:
    st.subheader("Subscripties per component")
    st.info(
        "Het aantal subscripties (Z) per component komt automatisch uit het "
        "werkelijke aantal klantlocaties; varieer prijs α en service level β^tar in "
        "de tabs Verwachte subscripties / Sensitivity om het verwachte aantal abonnees te zien. "
        "Een vaste override per component kun je hieronder bij 'IP / Levertijd / "
        "Z aanpassen' instellen.",
        icon="ℹ️",
    )

    st.divider()
    st.subheader("Overrides per artikelcode")
    st.caption("Z = aantal subscripties, IP = inkoopprijs (€), LT = levertijd (dagen). "
               "Laat een cel leeg om de Excel-waarde te gebruiken.")

    cfg.setdefault("ip_overrides", {})
    cfg.setdefault("lt_overrides", {})

    # Bouw gecombineerde tabel van alle codes met minstens één override
    alle_codes = sorted(
        set(cfg["n_klanten_overrides"]) |
        set(cfg["ip_overrides"]) |
        set(cfg["lt_overrides"])
    )
    override_rows = [
        {
            "Artikelcode": c,
            "N":           cfg["n_klanten_overrides"].get(c),
            "IP (€)":      cfg["ip_overrides"].get(c),
            "LT (dagen)":  cfg["lt_overrides"].get(c),
        }
        for c in alle_codes
    ]

    edited = st.data_editor(
        pd.DataFrame(override_rows) if override_rows else pd.DataFrame(
            columns=["Artikelcode", "N", "IP (€)", "LT (dagen)"]
        ),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Artikelcode": st.column_config.TextColumn("Artikelcode", required=True),
            "N":           st.column_config.NumberColumn("Z (subscripties)", min_value=1, step=1),
            "IP (€)":      st.column_config.NumberColumn("IP (€)", min_value=0.0, format="%.2f"),
            "LT (dagen)":  st.column_config.NumberColumn("LT (dagen)", min_value=1, step=1),
        },
        key="overrides_editor",
    )

    if st.button("💾 Opslaan overrides"):
        n_ov, ip_ov, lt_ov = {}, {}, {}
        for _, row in edited.iterrows():
            code = row.get("Artikelcode")
            if not code or pd.isna(code):
                continue
            code = str(code)
            if pd.notna(row["N"]):
                n_ov[code]  = int(row["N"])
            if pd.notna(row["IP (€)"]):
                ip_ov[code] = float(row["IP (€)"])
            if pd.notna(row["LT (dagen)"]):
                lt_ov[code] = int(row["LT (dagen)"])
        cfg["n_klanten_overrides"] = n_ov
        cfg["ip_overrides"]        = ip_ov
        cfg["lt_overrides"]        = lt_ov
        # Bewaar huidige overzicht_df als vorige snapshot vóór recompute
        if "overzicht_df" in st.session_state:
            st.session_state.overzicht_df_prev = st.session_state.overzicht_df.copy()
        sla_config_op(cfg)
        st.toast(f"Overrides opgeslagen — {len(n_ov)} Z, {len(ip_ov)} IP, {len(lt_ov)} LT.", icon="✅")
        st.session_state.pop("overzicht_df", None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 – COMPONENT TOEVOEGEN
# ─────────────────────────────────────────────────────────────────────────────

with tab_toevoegen:
    st.subheader("Nieuw component toevoegen")
    st.write("Gebruik dit voor componenten die nog niet in de Excel staan.")

    with st.form("form_toevoegen"):
        col1, col2 = st.columns(2)
        with col1:
            f_code  = st.text_input("Artikelcode *")
            f_descr = st.text_input("Omschrijving")
            f_lam   = st.number_input(
                "Lambda – vraag per jaar *",
                min_value=0.0001, value=1.0, step=0.1, format="%.4f",
            )
        with col2:
            f_lt = st.number_input(
                "Levertijd leverancier → BPA (dagen) *",
                min_value=1, value=30, step=1,
            )
            f_n = st.number_input(
                "Aantal subscripties (Z)",
                min_value=1, value=1, step=1,
            )
            f_ip = st.number_input(
                "Inkoopprijs (€)", min_value=0.0, value=0.0, step=10.0, format="%.2f",
            )
        submitted = st.form_submit_button("➕ Component opslaan")

    if submitted:
        if not f_code:
            st.error("Artikelcode is verplicht.")
        elif f_code in cfg["handmatige_componenten"]:
            st.warning(f"'{f_code}' bestaat al. Verwijder het eerst via het tabblad 'Component verwijderen'.")
        else:
            cfg["handmatige_componenten"][f_code] = {
                "descr":           f_descr,
                "lambda_per_jaar": float(f_lam),
                "lt_dagen":        int(f_lt),
                "n_klanten":       int(f_n),
                "ip":              float(f_ip),
            }
            sla_config_op(cfg)
            st.success(f"Component '{f_code}' toegevoegd.")

            # Preview berekende basisvoorraden
            lt_jr = int(f_lt) / 365
            preview = {
                f"s@{sl:.1%}": BPAOptimizationModel.inverse_service_level(sl, float(f_lam), lt_jr)
                for sl in SERVICE_LEVELS
            }
            st.write("**Berekende basisvoorraden voor dit component:**")
            st.dataframe(pd.DataFrame([preview]), use_container_width=False)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 – COMPONENT VERWIJDEREN
# ─────────────────────────────────────────────────────────────────────────────

with tab_verwijderen:
    st.subheader("Component verwijderen uit model")
    st.write("Handmatig toegevoegde componenten worden permanent verwijderd. "
             "Excel-componenten worden uitgesloten (en kunnen later weer worden teruggezet).")

    handmatig   = cfg["handmatige_componenten"]
    uitgesloten = cfg.setdefault("uitgesloten_componenten", [])

    # Gebruik dezelfde codes als in het Overzicht-tab (= classificatie-whitelist
    # toegepast, inclusief synthetische classificatie-rijen).
    _ov_df = st.session_state.get("overzicht_df")
    if _ov_df is None or _ov_df.empty:
        try:
            _ov_df = get_overzicht_df(cfg)
            st.session_state["overzicht_df"] = _ov_df
        except Exception:
            _ov_df = pd.DataFrame()

    if _ov_df is not None and not _ov_df.empty and "bron" in _ov_df.columns:
        excel_codes = [str(c) for c, b in zip(_ov_df.index, _ov_df["bron"])
                       if b in ("excel", "classificatie")]
    else:
        excel_codes = []

    # Alle actieve codes met bron
    opties = (
        [(c, "handmatig", handmatig[c].get("descr", "")) for c in handmatig if c not in uitgesloten] +
        [(c, "excel",     "") for c in excel_codes if c not in handmatig and c not in uitgesloten]
    )

    if not opties:
        st.info("Geen actieve componenten om te verwijderen.")
    else:
        keuze = st.selectbox(
            "Selecteer component",
            options=[c for c, _, _ in opties],
            format_func=lambda c: next(
                f"{c}  [{bron}]  {descr}" for code, bron, descr in opties if code == c
            ),
        )
        bron_keuze = next(bron for c, bron, _ in opties if c == keuze)
        if bron_keuze == "handmatig":
            v = handmatig[keuze]
            st.write(f"**{keuze}** (handmatig) &nbsp;|&nbsp; λ = {v['lambda_per_jaar']:.4f}/jr "
                     f"&nbsp;|&nbsp; LT = {v['lt_dagen']} d")
            st.warning("Dit component wordt permanent verwijderd.")
            if st.button("🗑️ Verwijder permanent", type="primary"):
                del cfg["handmatige_componenten"][keuze]
                sla_config_op(cfg)
                st.success(f"'{keuze}' verwijderd.")
                st.rerun()
        else:
            st.write(f"**{keuze}** (uit Excel) – wordt uitgesloten van berekeningen.")
            st.info("Het artikel blijft in de Excel staan maar telt niet meer mee in het model.")
            if st.button("🚫 Uitsluiten van model", type="primary"):
                if keuze not in uitgesloten:
                    uitgesloten.append(keuze)
                sla_config_op(cfg)
                st.success(f"'{keuze}' uitgesloten.")
                st.rerun()

    # Uitgesloten Excel-componenten terugzetten
    if uitgesloten:
        st.divider()
        st.subheader("Uitgesloten componenten terugzetten")
        terugzetten = st.selectbox(
            "Selecteer component om terug te zetten",
            options=uitgesloten,
            key="terugzetten_selectbox",
        )
        if st.button("↩️ Zet terug in model"):
            uitgesloten.remove(terugzetten)
            sla_config_op(cfg)
            st.success(f"'{terugzetten}' is weer actief.")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 – HISTORIEK
# ─────────────────────────────────────────────────────────────────────────────

with tab_historie:
    st.subheader("Historiek basisvoorraden")
    st.caption("Elke keer dat je een wijziging opslaat wordt automatisch een snapshot bewaard.")

    if not os.path.exists(HISTORY_PATH):
        st.info("Nog geen historiek beschikbaar. Sla een wijziging op om de eerste snapshot te maken.")
    else:
        with open(HISTORY_PATH, encoding="utf-8") as _f:
            history = json.load(_f)

        if not history:
            st.info("Nog geen snapshots.")
        else:
            # Bouw DataFrame op
            rows = []
            for h in history:
                row = {"Datum": h["datum"], "Z": h["n_klanten"], "# componenten": h["n_actief"]}
                row.update(h.get("totalen", {}))
                rows.append(row)
            hist_df = pd.DataFrame(rows).set_index("Datum")

            sl_cols = [c for c in hist_df.columns if c.startswith("s@")]

            # Grafiek
            if sl_cols:
                import matplotlib.pyplot as plt
                import matplotlib.ticker as ticker

                fig, ax = plt.subplots(figsize=(10, 4))
                for col in sl_cols:
                    ax.plot(hist_df.index, hist_df[col], marker="o", linewidth=2, label=col)
                    for x, y in zip(hist_df.index, hist_df[col]):
                        ax.annotate(str(int(y)), (x, y), textcoords="offset points",
                                    xytext=(0, 6), ha="center", fontsize=8)

                ax.set_xlabel("Update date", fontsize=11)
                ax.set_ylabel("Total base stock (units)", fontsize=11)
                ax.set_title("Total BPA base stock per update moment", fontsize=12)
                ax.legend(fontsize=9)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=30, ha="right")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Tabel
            st.write("**Snapshots:**")
            st.dataframe(hist_df.reset_index(), use_container_width=True)

            # Snapshot handmatig toevoegen (huidige staat)
            st.divider()
            if st.button("📸 Voeg snapshot toe van huidige staat"):
                from bpa_beheer import _sla_history_snapshot
                _sla_history_snapshot(cfg)
                st.success("Snapshot toegevoegd.")
                st.rerun()

    # ── Sensitivity grafieken ──────────────────────────────────────────────
    st.divider()
    st.subheader("Sensitivity grafieken")

    if "overzicht_df" not in st.session_state or st.session_state.overzicht_df.empty:
        st.info("Laad het overzicht (tabblad 📊) om de sensitivity grafieken te berekenen.")
    else:
        # Haal draaiknoppen op uit Kostenanalyse; gebruik defaults als nog niet berekend
        _kp = st.session_state.get('kosten_params', {})
        _ALPHA_DEF      = _kp.get('alpha',     0.15)
        _KAPPA_BPA_DEF  = _kp.get('kappa_bpa', 0.20)
        _KAPPA_C_DEF    = _kp.get('kappa_c',   0.25)

        st.caption(
            f"Vaste waarden buiten de gesweepte parameter: "
            f"α = **{_ALPHA_DEF:.0%}**, κ\\_BPA = **{_KAPPA_BPA_DEF:.0%}**, "
            f"κ\\_c = **{_KAPPA_C_DEF:.0%}**, N = standaard uit overzicht. "
            f"_(pas aan via tabblad 💰 Kostenanalyse)_"
        )

        _SL_SWEEP_S     = SERVICE_LEVELS
        _N_VALS         = [1, 2, 5, 10, 50]
        _ALPHA_SWEEP_S  = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
        _SL_ALPHA       = [sl for sl in SERVICE_LEVELS if sl >= 0.98]

        if st.button("📊 Bereken sensitivity grafieken"):
            _ov = st.session_state.overzicht_df
            _g1 = {n: [] for n in _N_VALS}
            _g2 = []
            _g3 = {sl: [] for sl in _SL_ALPHA}

            with st.spinner("Berekenen (kan even duren)…"):
                # Grafieken 1 & 2: sweep over service levels
                for _sl in _SL_SWEEP_S:
                    try:
                        _m2, _ = bouw_model_kosten(_ov, _ALPHA_DEF, _KAPPA_BPA_DEF, _KAPPA_C_DEF, _sl)
                        _g2.append({'sl': _sl, 'base': sum(_m2.calculate_base_stock_levels().values())})
                    except Exception:
                        _g2.append({'sl': _sl, 'base': None})
                    for _n in _N_VALS:
                        try:
                            _, _r1 = bouw_model_kosten(
                                _ov, _ALPHA_DEF, _KAPPA_BPA_DEF, _KAPPA_C_DEF, _sl,
                                n_klanten_override=_n,
                            )
                            _g1[_n].append({'sl': _sl, 'marge': _r1['bpa_margin']})
                        except Exception:
                            _g1[_n].append({'sl': _sl, 'marge': None})
                # Grafiek 3: sweep over alpha per SL ≥ 98%
                for _sl in _SL_ALPHA:
                    for _a in _ALPHA_SWEEP_S:
                        try:
                            _, _r3 = bouw_model_kosten(_ov, _a, _KAPPA_BPA_DEF, _KAPPA_C_DEF, _sl)
                            _g3[_sl].append({'alpha': _a, 'marge': _r3['bpa_margin']})
                        except Exception:
                            _g3[_sl].append({'alpha': _a, 'marge': None})

            st.session_state.sens_g1 = _g1
            st.session_state.sens_g2 = _g2
            st.session_state.sens_g3 = _g3

        if 'sens_g1' in st.session_state:
            import matplotlib.pyplot as _plt
            import matplotlib.ticker as _mt

            _COLORS5 = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#D32F2F']
            _COLORS4 = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
            _fmt_eur = _mt.FuncFormatter(lambda v, _: f'€{v:,.0f}')
            _fmt_sl  = _mt.FuncFormatter(lambda v, _: f'{v:.2f}%')

            # ── Grafiek 1: service level vs marge per N ────────────────────
            _fig1, _ax1 = _plt.subplots(figsize=(10, 5))
            for _n, _col in zip(_N_VALS, _COLORS5):
                _pts = [(r['sl']*100, r['marge'])
                        for r in st.session_state.sens_g1[_n] if r['marge'] is not None]
                if _pts:
                    _ax1.plot([p[0] for p in _pts], [p[1] for p in _pts],
                              marker='o', linewidth=2, color=_col, label=f'N = {_n}')
            _ax1.axhline(0, color='grey', linewidth=0.8)
            _ax1.set_xlabel('Service level (%)', fontsize=11)
            _ax1.set_ylabel('Annual margin (€)', fontsize=11)
            _ax1.set_title(
                f'Margin vs. service level  '
                f'(α = {_ALPHA_DEF:.0%}, κ_BPA = {_KAPPA_BPA_DEF:.0%})',
                fontsize=12,
            )
            _ax1.yaxis.set_major_formatter(_fmt_eur)
            _ax1.xaxis.set_major_formatter(_fmt_sl)
            _ax1.set_xticks([sl*100 for sl in _SL_SWEEP_S])
            _ax1.legend(fontsize=9)
            _ax1.grid(True, alpha=0.3)
            _plt.setp(_ax1.get_xticklabels(), rotation=25, ha='right')
            _fig1.tight_layout()
            st.pyplot(_fig1)
            _plt.close(_fig1)

            # ── Grafiek 2: service level vs basisvoorraad ──────────────────
            _fig2, _ax2 = _plt.subplots(figsize=(10, 4))
            _pts2 = [(r['sl']*100, r['base'])
                     for r in st.session_state.sens_g2 if r['base'] is not None]
            if _pts2:
                _ax2.plot([p[0] for p in _pts2], [p[1] for p in _pts2],
                          marker='s', linewidth=2, color='#FF9800', label='Total S*')
                for _xv, _yv in _pts2:
                    _ax2.annotate(str(int(_yv)), (_xv, _yv),
                                  textcoords='offset points', xytext=(0, 7),
                                  ha='center', fontsize=9)
            _ax2.set_xlabel('Service level (%)', fontsize=11)
            _ax2.set_ylabel('Total base stock (units)', fontsize=11)
            _ax2.set_title(
                f'Base stock vs. service level  '
                f'(α = {_ALPHA_DEF:.0%}, N = standard)',
                fontsize=12,
            )
            _ax2.xaxis.set_major_formatter(_fmt_sl)
            _ax2.set_xticks([sl*100 for sl in _SL_SWEEP_S])
            _ax2.yaxis.set_major_locator(_mt.MaxNLocator(integer=True))
            _ax2.legend(fontsize=9)
            _ax2.grid(True, alpha=0.3)
            _plt.setp(_ax2.get_xticklabels(), rotation=25, ha='right')
            _fig2.tight_layout()
            st.pyplot(_fig2)
            _plt.close(_fig2)

            # ── Grafiek 3: alpha vs marge per service level ────────────────
            _fig3, _ax3 = _plt.subplots(figsize=(10, 5))
            for _sl3, _col3 in zip(_SL_ALPHA, _COLORS4):
                _pts3 = [(r['alpha']*100, r['marge'])
                         for r in st.session_state.sens_g3[_sl3] if r['marge'] is not None]
                if _pts3:
                    _ax3.plot([p[0] for p in _pts3], [p[1] for p in _pts3],
                              marker='o', linewidth=2, color=_col3, label=f'SL = {_sl3:.1%}')
            _ax3.axhline(0, color='grey', linewidth=0.8)
            _ax3.set_xlabel('Subscription rate α (%)', fontsize=11)
            _ax3.set_ylabel('Annual margin (€)', fontsize=11)
            _ax3.set_title(
                f'Margin vs. subscription rate  '
                f'(κ_BPA = {_KAPPA_BPA_DEF:.0%}, N = standard)',
                fontsize=12,
            )
            _ax3.yaxis.set_major_formatter(_fmt_eur)
            _ax3.xaxis.set_major_formatter(_mt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
            _ax3.set_xticks([a*100 for a in _ALPHA_SWEEP_S])
            _ax3.legend(fontsize=9)
            _ax3.grid(True, alpha=0.3)
            _plt.setp(_ax3.get_xticklabels(), rotation=25, ha='right')
            _fig3.tight_layout()
            st.pyplot(_fig3)
            _plt.close(_fig3)

        # ── Haalbaarheid BPA per (N, SL) – heatmap ────────────────────────────────
        st.divider()
        st.subheader("Haalbaarheid BPA per (Z, serviceniveau)")
        st.caption(
            "Groen = BPA is haalbaar (marge ≥ 0), rood = niet haalbaar. "
            "α wordt overgenomen uit tabblad 💰 Kostenanalyse; κ_BPA en κ_c idem."
        )

        _N_NSL_VALS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]

        if st.button("📊 Bereken haalbaarheid (N × SL)"):
            _kp_nsl = st.session_state.get('kosten_params', {})
            _a_nsl  = _kp_nsl.get('alpha',     0.15)
            _kb_nsl = _kp_nsl.get('kappa_bpa', 0.20)
            _kc_nsl = _kp_nsl.get('kappa_c',   0.25)

            _nsl_grid = {}
            with st.spinner("Berekenen haalbaarheid (N × SL)…"):
                for _n_nsl in _N_NSL_VALS:
                    _nsl_grid[_n_nsl] = {}
                    for _sl_nsl in SERVICE_LEVELS:
                        try:
                            _, _r_nsl = bouw_model_kosten(
                                st.session_state.overzicht_df,
                                _a_nsl, _kb_nsl, _kc_nsl, _sl_nsl,
                                n_klanten_override=_n_nsl,
                            )
                            _nsl_grid[_n_nsl][_sl_nsl] = {
                                'feasible': _r_nsl['feasible'],
                                'margin':   _r_nsl['bpa_margin'],
                            }
                        except Exception:
                            _nsl_grid[_n_nsl][_sl_nsl] = {'feasible': False, 'margin': None}

            st.session_state.sens_nsl_grid  = _nsl_grid
            st.session_state.sens_nsl_alpha = _a_nsl
            st.session_state.sens_nsl_kb    = _kb_nsl

        if 'sens_nsl_grid' in st.session_state:
            import matplotlib.pyplot as _plt_nsl
            import matplotlib.colors as _mcolors_nsl
            import numpy as _np_nsl

            _grid   = st.session_state.sens_nsl_grid
            _a_lbl  = st.session_state.sens_nsl_alpha
            _kb_lbl = st.session_state.sens_nsl_kb
            _n_std_nsl = representatieve_z()

            _rows_nsl = SERVICE_LEVELS       # y-as
            _cols_nsl = _N_NSL_VALS          # x-as

            # Bouw matrices: haalbaarheid (0/1) en genormaliseerde marge
            _feas_mat = _np_nsl.zeros((len(_rows_nsl), len(_cols_nsl)))
            _marg_mat = _np_nsl.full((len(_rows_nsl), len(_cols_nsl)), float('nan'))

            for _ci, _n_v in enumerate(_cols_nsl):
                for _ri, _sl_v in enumerate(_rows_nsl):
                    _cell = _grid.get(_n_v, {}).get(_sl_v, {})
                    _feas_mat[_ri, _ci] = 1.0 if _cell.get('feasible') else 0.0
                    if _cell.get('margin') is not None:
                        _marg_mat[_ri, _ci] = _cell['margin']

            # Kleurschaal: rood → geel → groen via marge-waarden
            _valid = _marg_mat[~_np_nsl.isnan(_marg_mat)]
            if len(_valid) > 0:
                _abs_max = max(abs(_valid.min()), abs(_valid.max()), 1)
            else:
                _abs_max = 1
            _norm_nsl = _mcolors_nsl.TwoSlopeNorm(
                vmin=-_abs_max, vcenter=0, vmax=_abs_max
            )

            _fig_nsl, _ax_nsl = _plt_nsl.subplots(figsize=(13, 5))
            _im_nsl = _ax_nsl.imshow(
                _marg_mat, aspect='auto',
                cmap='RdYlGn', norm=_norm_nsl,
                interpolation='nearest',
            )
            _plt_nsl.colorbar(_im_nsl, ax=_ax_nsl, label='BPA margin (€)', fraction=0.03, pad=0.02)

            # Annotaties per cel
            for _ri, _sl_v in enumerate(_rows_nsl):
                for _ci, _n_v in enumerate(_cols_nsl):
                    _cell = _grid.get(_n_v, {}).get(_sl_v, {})
                    _feas = _cell.get('feasible', False)
                    _mg   = _cell.get('margin')
                    _sym  = '✓' if _feas else '✗'
                    _tc   = '#1a5c1a' if _feas else '#7a0000'
                    _ax_nsl.text(_ci, _ri, _sym,
                                 ha='center', va='center' if _mg is None else 'bottom',
                                 fontsize=13, color=_tc, fontweight='bold')
                    if _mg is not None:
                        _ax_nsl.text(_ci, _ri + 0.28, f'€{_mg:,.0f}',
                                     ha='center', va='center', fontsize=6.5, color=_tc)

            # Assen
            _ax_nsl.set_xticks(range(len(_cols_nsl)))
            _ax_nsl.set_xticklabels([str(n) for n in _cols_nsl], fontsize=9)
            _ax_nsl.set_yticks(range(len(_rows_nsl)))
            _ax_nsl.set_yticklabels([f'{sl:.1%}' for sl in _rows_nsl], fontsize=9)
            _ax_nsl.set_xlabel('Number of subscriptions (Z)', fontsize=11)
            _ax_nsl.set_ylabel('Service level', fontsize=11)
            _ax_nsl.set_title(
                f'BPA feasibility per (Z, service level)  '
                f'(α = {_a_lbl:.0%}, κ_BPA = {_kb_lbl:.0%})',
                fontsize=12,
            )

            # Markeer huidige N
            try:
                _ni_std = min(range(len(_cols_nsl)),
                              key=lambda k: abs(_cols_nsl[k] - _n_std_nsl))
                _ax_nsl.axvline(_ni_std, color='black', linewidth=2.0, linestyle=':')
                _ax_nsl.text(_ni_std + 0.15, -0.7, f'Z={_n_std_nsl}',
                             fontsize=8, color='black')
            except Exception:
                pass

            _fig_nsl.tight_layout()
            st.pyplot(_fig_nsl)
            _plt_nsl.close(_fig_nsl)

        # ── Investering vs. N ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Investering vs. aantal subscripties")
        st.caption(
            "Totale voorraadwaarde (Σ S\u002a × inkoopprijs) als functie van het aantal "
            "subscripties per service level. Toont hoeveel kapitaal BPA in voorraad "
            "moet investeren naarmate het klantenbestand groeit. De x-as toont het "
            "TOTALE aantal subscripties over alle componenten en start bij de som van "
            "de verwachte sub-aantallen (E[Z]); alle componenten schalen van daaruit "
            "proportioneel mee omhoog."
        )

        # Baseline per component = het geconfigureerde n_klanten, dat de
        # verwachte E[Z_i(α,X)] weergeeft zodra die via de tab Verwachte
        # subscripties is doorgezet. De x-as toont het TOTAAL aantal
        # subscripties over alle componenten (= som van de baselines bij
        # factor 1.0); alle componenten schalen proportioneel mee.
        _sim_base_inv = {}
        # Groeifactoren: 20 punten van 1.0x (huidig totaal) tot 2.9x in stappen van 0.1.
        _INV_FACTORS = [round(1.0 + 0.1 * _k, 1) for _k in range(20)]
        _COLORS_INV = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']

        if st.button("📊 Bereken investering vs. totaal subs"):
            _ov_inv = st.session_state.overzicht_df.reset_index()
            # Verzamel per component: lambda per subscriptie, baseline-subs, LT, IP, VP, code
            _comp_inv = []
            for _, _ri in _ov_inv.iterrows():
                _ni = float(_ri.get('n_klanten', 0) or 0)
                _li = float(_ri.get('lambda_jr', 0) or 0)
                _lt = float(_ri.get('LT_dagen', 0) or 0)
                _ip = float(_ri.get('IP', 0) or 0)
                _vp = float(_ri.get('VP', 0) or 0)
                if _ni > 0 and _li > 0 and _lt > 0:
                    _code = str(_ri.get('Code', ''))
                    _comp_inv.append({
                        'code':        _code,
                        'descr':       str(_ri.get('Descr', '')),
                        'lam_per_sub': _li / _ni,
                        'n_base':      float(_sim_base_inv.get(_code, _ni)),
                        'lt_jr':       _lt / 365,
                        'ip':          _ip,
                        'vp':          _vp,
                    })

            # Totaal subs bij factor 1.0 = som van de (verwachte) baselines.
            _T0_inv = sum(_c['n_base'] for _c in _comp_inv)

            _inv_results = {sl: [] for sl in SERVICE_LEVELS}
            # Per-component resultaten voor top-5 grafiek (alle SL's)
            _sl_top = st.session_state.get('kosten_params', {}).get('service_level', 0.990)
            _inv_per_comp = {sl: {c['code']: [] for c in _comp_inv} for sl in SERVICE_LEVELS}

            with st.spinner("Berekenen investering vs. totaal subs…"):
                for _f_inv in _INV_FACTORS:
                    _tot_subs = int(round(_T0_inv * _f_inv))   # x-waarde: totaal subscripties
                    for _sl_inv in SERVICE_LEVELS:
                        _totaal = sum(
                            BPAOptimizationModel.inverse_service_level(
                                _sl_inv, _c['lam_per_sub'] * _c['n_base'] * _f_inv, _c['lt_jr']
                            ) * _c['ip']
                            for _c in _comp_inv
                        )
                        _inv_results[_sl_inv].append({'n': _tot_subs, 'inv': _totaal})
                    # Per-component per SL (voor top-5/top-10 grafiek)
                    for _c in _comp_inv:
                        for _sl_c in SERVICE_LEVELS:
                            _s = BPAOptimizationModel.inverse_service_level(
                                _sl_c, _c['lam_per_sub'] * _c['n_base'] * _f_inv, _c['lt_jr']
                            )
                            _inv_per_comp[_sl_c][_c['code']].append({'n': _tot_subs, 'inv': _s * _c['ip']})

            # Top 5 / Top 10 duurste componenten op VP
            _top5_codes  = sorted(_comp_inv, key=lambda c: c['vp'], reverse=True)[:5]
            _top10_codes = sorted(_comp_inv, key=lambda c: c['vp'], reverse=True)[:10]

            st.session_state.sens_inv        = _inv_results
            st.session_state.sens_inv_comp   = _inv_per_comp
            st.session_state.sens_inv_top5   = _top5_codes
            st.session_state.sens_inv_top10  = _top10_codes
            st.session_state.sens_inv_sl_top = _sl_top
            st.session_state.sens_inv_t0     = int(round(_T0_inv))

        if 'sens_inv' in st.session_state:
            import matplotlib.pyplot as _plt_inv
            import matplotlib.ticker as _mt_inv

            _inv_d    = st.session_state.sens_inv
            _tot0_inv = int(st.session_state.get('sens_inv_t0', 0))
            _x_ticks_inv = [r['n'] for r in _inv_d[SERVICE_LEVELS[0]]]
            _fmt_inv  = _mt_inv.FuncFormatter(lambda v, _: f'€{v:,.0f}')

            _fig_inv, _ax_inv = _plt_inv.subplots(figsize=(11, 5))
            for _sl_inv, _col_inv in zip(SERVICE_LEVELS, _COLORS_INV):
                _pts_inv = [(r['n'], r['inv']) for r in _inv_d[_sl_inv] if r['inv'] is not None]
                if _pts_inv:
                    _xi, _yi = zip(*_pts_inv)
                    _ax_inv.plot(_xi, _yi, marker='o', linewidth=2,
                                 color=_col_inv, label=f'SL = {_sl_inv:.1%}')

            _ax_inv.axvline(_tot0_inv, color='black', linewidth=1.0, linestyle=':',
                            label=f'Total subs (sim) = {_tot0_inv}')
            _ax_inv.set_xlabel('Total number of subscriptions (all components)', fontsize=11)
            _ax_inv.set_ylabel('Total inventory value (€)', fontsize=11)
            _ax_inv.set_title(
                'Required investment in base stock vs. total number of subscriptions',
                fontsize=12,
            )
            _ax_inv.yaxis.set_major_formatter(_fmt_inv)
            _ax_inv.set_xticks(_x_ticks_inv)
            _plt_inv.setp(_ax_inv.get_xticklabels(), rotation=30, ha='right')
            _ax_inv.legend(fontsize=9)
            _ax_inv.grid(True, alpha=0.3)
            _fig_inv.tight_layout()
            st.pyplot(_fig_inv)
            _plt_inv.close(_fig_inv)

            # Tabel: investering per totaal aantal subs en SL
            _inv_tbl_rows = []
            for _n_v in _x_ticks_inv:
                _row_t = {'Totaal subs': _n_v}
                for _sl_v in SERVICE_LEVELS:
                    _pts = [r for r in _inv_d[_sl_v] if r['n'] == _n_v]
                    _row_t[f'SL {_sl_v:.1%}'] = f"€{_pts[0]['inv']:,.0f}" if _pts else '—'
                _inv_tbl_rows.append(_row_t)
            st.dataframe(pd.DataFrame(_inv_tbl_rows).set_index('Totaal subs'), use_container_width=False)

            # ── Top-5 duurste componenten per VP ──────────────────────────
            if 'sens_inv_top5' in st.session_state:
                _top5    = st.session_state.sens_inv_top5
                _comp_d  = st.session_state.sens_inv_comp
                _sl_lbl  = st.session_state.sens_inv_sl_top

                _COLORS_TOP5 = ['#D32F2F', '#F57C00', '#FBC02D', '#388E3C', '#1976D2']
                st.subheader("Top 5 duurste componenten (VP) — investering vs. totaal subs (per service level)")
                st.caption(
                    "Gesommeerde investeringswaarde (S\u002a × IP) van de top 5 duurste componenten "
                    "(op verkoopprijs) als functie van het totaal aantal subscripties, per service level."
                )

                # Haal x-waarden op uit de data
                _comp_d_sl0 = _comp_d.get(SERVICE_LEVELS[0], {})
                _x5_vals = [p['n'] for p in _comp_d_sl0.get(_top5[0]['code'], [])] if _top5 else []

                if _x5_vals:
                    _fig_t5, _ax_t5 = _plt_inv.subplots(figsize=(11, 5))

                    for _sl_t5, _col_t5, _ls_t5 in zip(
                            SERVICE_LEVELS, _COLORS_INV, ['-', '--', '-.', ':']):
                        _cd_sl = _comp_d.get(_sl_t5, {})
                        _tot_sl = [
                            sum(
                                next((p['inv'] for p in _cd_sl.get(_c5['code'], [])
                                      if p['n'] == _nv), 0)
                                for _c5 in _top5
                            )
                            for _nv in _x5_vals
                        ]
                        _ax_t5.plot(_x5_vals, _tot_sl, color=_col_t5, marker='o',
                                    linewidth=2.0, linestyle=_ls_t5,
                                    label=f'SL {_sl_t5:.1%}')

                    _ax_t5.axvline(_tot0_inv, color='black', linewidth=1.0, linestyle=':',
                                   label=f'Total subs (sim) = {_tot0_inv}')
                    _ax_t5.set_xlabel('Total number of subscriptions (all components)', fontsize=11)
                    _ax_t5.set_ylabel('Summed investment value top 5 (€)', fontsize=11)
                    _ax_t5.set_title(
                        'Top 5 most expensive components (VP): summed investment vs. total subs per SL',
                        fontsize=12,
                    )
                    _ax_t5.yaxis.set_major_formatter(_fmt_inv)
                    _ax_t5.set_xticks(_x5_vals)
                    _plt_inv.setp(_ax_t5.get_xticklabels(), rotation=30, ha='right')
                    _ax_t5.legend(fontsize=9, loc='upper left')
                    _ax_t5.grid(True, alpha=0.3)
                    _fig_t5.tight_layout()
                    st.pyplot(_fig_t5)
                    _plt_inv.close(_fig_t5)

            # ── Top-10 duurste componenten — gesommeerde lijnen per SL ──────
            if 'sens_inv_top10' in st.session_state:
                import matplotlib as _mpl_inv
                _top10   = st.session_state.sens_inv_top10
                _comp_d  = st.session_state.sens_inv_comp

                _comp_d_sl0_t10 = _comp_d.get(SERVICE_LEVELS[0], {})
                _x10_sum_vals = [p['n'] for p in _comp_d_sl0_t10.get(_top10[0]['code'], [])] if _top10 else []

                st.subheader("Top 10 duurste componenten (VP) — gesommeerde investering vs. totaal subs (per service level)")
                st.caption(
                    "Gesommeerde investeringswaarde (S\u002a × IP) van de top 10 duurste componenten "
                    "(op verkoopprijs) als functie van het totaal aantal subscripties, per service level."
                )

                if _x10_sum_vals:
                    _fig_t10s, _ax_t10s = _plt_inv.subplots(figsize=(11, 5))
                    for _sl_t10s, _col_t10s, _ls_t10s in zip(
                            SERVICE_LEVELS, _COLORS_INV, ['-', '--', '-.', ':']):
                        _cd10s = _comp_d.get(_sl_t10s, {})
                        _tot10s = [
                            sum(
                                next((p['inv'] for p in _cd10s.get(_c10['code'], [])
                                      if p['n'] == _nv), 0)
                                for _c10 in _top10
                            )
                            for _nv in _x10_sum_vals
                        ]
                        _ax_t10s.plot(_x10_sum_vals, _tot10s, color=_col_t10s, marker='o',
                                      linewidth=2.0, linestyle=_ls_t10s,
                                      label=f'SL {_sl_t10s:.1%}')
                    _ax_t10s.axvline(_tot0_inv, color='black', linewidth=1.0, linestyle=':',
                                     label=f'Total subs (sim) = {_tot0_inv}')
                    _ax_t10s.set_xlabel('Total number of subscriptions (all components)', fontsize=11)
                    _ax_t10s.set_ylabel('Summed investment value top 10 (€)', fontsize=11)
                    _ax_t10s.set_title(
                        'Top 10 most expensive components (VP): summed investment vs. total subs per SL',
                        fontsize=12,
                    )
                    _ax_t10s.yaxis.set_major_formatter(_fmt_inv)
                    _ax_t10s.set_xticks(_x10_sum_vals)
                    _plt_inv.setp(_ax_t10s.get_xticklabels(), rotation=30, ha='right')
                    _ax_t10s.legend(fontsize=9, loc='upper left')
                    _ax_t10s.grid(True, alpha=0.3)
                    _fig_t10s.tight_layout()
                    st.pyplot(_fig_t10s)
                    _plt_inv.close(_fig_t10s)

                # ── Top-10 duurste componenten — individuele lijnen ──────────
                _sl_opts_t10 = [f'SL {s:.1%}' for s in SERVICE_LEVELS]
                _sl_sel_t10  = st.selectbox(
                    'Service level voor top-10 grafiek',
                    _sl_opts_t10,
                    index=1,
                    key='top10_sl_select',
                )
                _sl_val_t10 = SERVICE_LEVELS[_sl_opts_t10.index(_sl_sel_t10)]

                st.subheader("Top 10 duurste componenten (VP) — investering per component vs. totaal subs")
                st.caption(
                    "Investeringswaarde (S\u002a \u00d7 IP) per component als functie van het totaal aantal subscripties "
                    "voor het geselecteerde service level."
                )

                _cd10_sl = _comp_d.get(_sl_val_t10, {})
                _x10_vals = [p['n'] for p in _cd10_sl.get(_top10[0]['code'], [])] if _top10 else []

                if _x10_vals:
                    _cmap10  = _mpl_inv.colormaps['tab10']
                    _fig_t10, _ax_t10 = _plt_inv.subplots(figsize=(12, 5))

                    for _ci, _c10 in enumerate(_top10):
                        _pts10 = [p['inv'] for p in _cd10_sl.get(_c10['code'], [])]
                        if _pts10:
                            _lbl10 = f"{_c10['code']} – {_c10.get('descr', '')[:25]}"
                            _ax_t10.plot(
                                _x10_vals, _pts10,
                                color=_cmap10(_ci / 10),
                                marker='o', linewidth=1.8, markersize=5,
                                label=_lbl10,
                            )

                    _ax_t10.axvline(_tot0_inv, color='black', linewidth=1.0, linestyle=':',
                                    label=f'Total subs (sim) = {_tot0_inv}')
                    _ax_t10.set_xlabel('Total number of subscriptions (all components)', fontsize=11)
                    _ax_t10.set_ylabel('Investment value per component (€)', fontsize=11)
                    _ax_t10.set_title(
                        f'Top 10 most expensive components — investment vs. total subs  ({_sl_sel_t10})',
                        fontsize=12,
                    )
                    _ax_t10.yaxis.set_major_formatter(_fmt_inv)
                    _ax_t10.set_xticks(_x10_vals)
                    _plt_inv.setp(_ax_t10.get_xticklabels(), rotation=30, ha='right')
                    _ax_t10.legend(fontsize=8, loc='upper left', ncol=2)
                    _ax_t10.grid(True, alpha=0.3)
                    _fig_t10.tight_layout()
                    st.pyplot(_fig_t10)
                    _plt_inv.close(_fig_t10)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 – KOSTENANALYSE
# ─────────────────────────────────────────────────────────────────────────────

with tab_kosten:
    st.subheader("Kostenanalyse BPA")
    st.caption(
        "Berekent BPA-kosten, omzet, marge en α-interval per component "
        "op basis van het huidige overzicht en de gekozen draaiknoppen."
    )

    if "overzicht_df" not in st.session_state or st.session_state.overzicht_df.empty:
        st.warning("Laad eerst het overzicht via het tabblad 📊 Overzicht.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            k_alpha = st.number_input(
                "α (abonnementstarief, %)",
                min_value=1.0, max_value=50.0, value=15.0, step=1.0, format="%.0f",
                help="Abonnementsprijs als percentage van verkoopprijs",
            ) / 100
        with col_b:
            k_kappa_bpa = st.number_input(
                "κ_BPA (%)",
                min_value=1.0, max_value=100.0, value=20.0, step=1.0, format="%.0f",
                help="κ_BPA = financiering + opslag + obsolescence (BPA)",
            ) / 100
        with col_c:
            k_kappa_c = st.number_input(
                "κ_c (%)",
                min_value=1.0, max_value=100.0, value=25.0, step=1.0, format="%.0f",
                help="κ_c = financiering + opslag + obsolescence (klant)",
            ) / 100
        with col_d:
            k_sl = st.selectbox(
                "Service level",
                options=SERVICE_LEVELS,
                index=SERVICE_LEVELS.index(0.990) if 0.990 in SERVICE_LEVELS else 0,
                format_func=lambda v: f"{v:.1%}",
            )

        _kost_ad1, _kost_ad2 = st.columns(2)
        with _kost_ad1:
            k_q_eq = st.number_input(
                "q_eq (adoptie bij pariteit)",
                min_value=0.01, max_value=0.99,
                value=float(st.session_state.get("subsim_q_eq", 0.55)),
                step=0.05, format="%.2f", key="kost_q_eq",
                help="Adoptiekans bij kostenpariteit α = κ_c.",
            )
        with _kost_ad2:
            k_beta_r = st.number_input(
                "η_r (kostenratio-gevoeligheid)",
                min_value=0.0, max_value=20.0,
                value=float(st.session_state.get("subsim_beta_r", 1.0)),
                step=0.1, format="%.2f", key="kost_beta_r",
                help="Gevoeligheid adoptie voor kostenratio ln(κ_c/α).",
            )

        if st.button("💰 Bereken kosten"):
            with st.spinner("Kostenmodel berekenen…"):
                try:
                    # Adoptie-bewuste overzicht: vervang n_klanten/lambda_jr door Z_i(α)
                    _k_ov = st.session_state.overzicht_df.copy()
                    try:
                        _k_excel_src = st.session_state.get("subsim_upload") or (
                            SUBSCRIPTIES_PATH if os.path.exists(SUBSCRIPTIES_PATH) else None)
                        if _k_excel_src is not None:
                            _k_n_mi = _cached_aantal_klanten(
                                _file_mtime(_k_excel_src if isinstance(_k_excel_src, str) else ""),
                                upload=_k_excel_src,
                            )
                            if not _k_n_mi.empty:
                                _k_q    = adoptie_kans(k_alpha, k_kappa_c, k_q_eq, k_beta_r)
                                _k_base = _k_ov.loc[_k_ov.index.isin(_k_n_mi.index)].copy()
                                _k_ez   = _k_n_mi.reindex(_k_base.index).fillna(0.0) * _k_q
                                _k_lpc  = (_k_base["lambda_jr"] / _k_base["n_klanten"].replace(0, np.nan)).fillna(0.0)
                                _k_base["n_klanten"] = _k_ez.round().clip(lower=0).astype(int)
                                _k_base["lambda_jr"] = (_k_ez * _k_lpc).values
                                _k_ov = _k_base
                    except Exception:
                        pass  # fallback op origineel overzicht
                    _m, _r = bouw_model_kosten(
                        _k_ov,
                        alpha=k_alpha,
                        kappa_bpa=k_kappa_bpa,
                        kappa_c=k_kappa_c,
                        service_level=k_sl,
                    )
                    st.session_state.kosten_result = (_m, _r)
                    st.session_state.kosten_params = {
                        'alpha': k_alpha, 'kappa_bpa': k_kappa_bpa,
                        'kappa_c': k_kappa_c, 'service_level': k_sl,
                    }
                except Exception as _e:
                    st.error(f"Fout bij berekening: {_e}")

        if "kosten_result" in st.session_state:
            _m, _r = st.session_state.kosten_result
            _iv = _r['alpha_intervals']
            _p  = st.session_state.kosten_params

            # ── Samenvatting ───────────────────────────────────────────────
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Haalbaar",    "✓ JA"  if _r['feasible'] else "✗ NEE")
            _c2.metric("Totale omzet",  f"€ {_r['total_revenue']:,.0f}")
            _c3.metric("BPA kosten",    f"€ {_r['bpa_costs']:,.0f}")
            _c4.metric("Marge",         f"€ {_r['bpa_margin']:+,.0f}")

            _al = _iv['universal_alpha_L']
            _au = _iv['universal_alpha_U']
            if _al is not None:
                st.info(
                    f"Universeel α-interval: **[{_al:.4%} – {_au:.4%}]**  "
                    f"{'✓ Haalbaar' if _iv['universal_feasible'] else '✗ Niet haalbaar'}"
                )

            # ── Per-component kosten tabel ─────────────────────────────────
            st.subheader("Kosten per component")
            _det = _m.calculate_detailed_bpa_costs()
            _bsl = _m.calculate_base_stock_levels()
            _per = _iv['per_component']
            _lt  = _m.parameters['lead_time']

            _rows = []
            for _code in _m.sets['spare_parts']:
                _d = _det[_code]
                _pc = _per.get(_code, {})
                _al = _pc.get('alpha_L')
                _au = _pc.get('alpha_U')
                _ok = (
                    _al is not None and _au is not None
                    and _al <= _p['alpha'] <= _au
                )
                _rows.append({
                    'Code':       _code,
                    'S*':         _bsl.get(_code, 0),
                    'Λ_BPA':      round(_d['demand'], 4),
                    'μ=Λ·L':      round(_d['demand'] * _lt.get(_code, 0), 4),
                    'C_BPA (€)':  round(_d['total'], 2),
                    'Omzet (€)':  round(_r['revenue_by_part'].get(_code, 0), 2),
                    'Marge (€)':  round(_r['revenue_by_part'].get(_code, 0) - _d['total'], 2),
                    'α_L,i':      f"{_al:.3%}" if _al is not None else '—',
                    'α_U,i':      f"{_au:.3%}" if _au is not None else '—',
                    'OK':         '✓' if _ok else '✗',
                })
            _tbl = pd.DataFrame(_rows).set_index('Code')

            st.dataframe(
                _tbl.style.format({
                    'S*':        '{:.0f}',
                    'Λ_BPA':    '{:.4f}',
                    'μ=Λ·L':    '{:.4f}',
                    'C_BPA (€)': '€ {:,.2f}',
                    'Omzet (€)': '€ {:,.2f}',
                    'Marge (€)': '€ {:+,.2f}',
                }),
                use_container_width=True,
                height=420,
            )
            st.write(
                f"**Totaal:** S\\* = {int(_tbl['S*'].sum())}  |  "
                f"C\\_BPA = € {_tbl['C_BPA (€)'].sum():,.2f}  |  "
                f"Omzet = € {_tbl['Omzet (€)'].sum():,.2f}  |  "
                f"Marge = € {_tbl['Marge (€)'].sum():+,.2f}"
            )

            # ── Klantbesparingen ───────────────────────────────────────────
            with st.expander("Klantbesparingen"):
                _klant_rows = [
                    {
                        'Klant':              _cust,
                        'Eigen kosten (€)':   b['self_stocking_cost'],
                        'BPA abonnement (€)': b['bpa_service_cost'],
                        'Besparing (€)':      b['savings'],
                        'Voordeel':           '✓' if b['benefits'] else '✗',
                    }
                    for _cust, b in _r['customer_benefits'].items()
                ]
                st.dataframe(
                    pd.DataFrame(_klant_rows).set_index('Klant').style.format({
                        'Eigen kosten (€)':   '€ {:,.2f}',
                        'BPA abonnement (€)': '€ {:,.2f}',
                        'Besparing (€)':      '€ {:+,.2f}',
                    }),
                    use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────────────────────────
#  TAB 8 – SUBSCRIPTIEDREMPEL
# ─────────────────────────────────────────────────────────────────────────────────

with tab_drempel:
    st.subheader("Subscriptiedrempel per component")
    st.caption(
        "Per component: hoeveel extra subscripties zijn er nodig voordat S\u002a met 1 stijgt? "
        "Aanname: λ schaalt lineair met Z (λ = Z × λ_huidig / Z_huidig). "
        "Van toepassing op MTBF-gebaseerde componenten."
    )

    if "overzicht_df" not in st.session_state or st.session_state.overzicht_df.empty:
        st.warning("Laad eerst het overzicht via het tabblad 📊 Overzicht.")
    else:
        _df_ov = st.session_state.overzicht_df.copy().reset_index()

        _sl_d = st.selectbox(
            "Service level",
            options=SERVICE_LEVELS,
            index=SERVICE_LEVELS.index(0.990) if 0.990 in SERVICE_LEVELS else 0,
            format_func=lambda v: f"{v:.1%}",
            key="drempel_sl",
        )
        _sl_col = f"s@{_sl_d:.1%}"

        # ── Adoptie-bewuste parameters: bereken Z_i(α) als huidig niveau ──
        _drm_c1, _drm_c2, _drm_c3 = st.columns(3)
        with _drm_c1:
            _drm_alpha = st.number_input(
                "α (abonnementstarief)",
                min_value=0.001, max_value=1.0,
                value=float(st.session_state.get("kosten_params", {}).get("alpha", 0.15)),
                step=0.01, format="%.3f", key="drm_alpha",
                help="Prijspercentage α waarvoor Z_i(α) = M_i·q(α) wordt berekend.",
            )
        with _drm_c2:
            _drm_q_eq = st.number_input(
                "q_eq (adoptie bij pariteit)",
                min_value=0.01, max_value=0.99,
                value=float(st.session_state.get("subsim_q_eq", 0.55)),
                step=0.05, format="%.2f", key="drm_q_eq",
            )
        with _drm_c3:
            _drm_beta_r = st.number_input(
                "η_r (kostenratio-gevoeligheid)",
                min_value=0.0, max_value=20.0,
                value=float(st.session_state.get("subsim_beta_r", 1.0)),
                step=0.1, format="%.2f", key="drm_beta_r",
            )
        _drm_kappa_c = float(st.session_state.get("kosten_params", {}).get("kappa_c", 0.25))
        _drm_q = adoptie_kans(_drm_alpha, _drm_kappa_c, _drm_q_eq, _drm_beta_r)

        # Laad M_i (cached) voor Z_i(α) = M_i·q(α)
        _drm_n_mi = pd.Series(dtype=float)
        try:
            _drm_excel_src = st.session_state.get("subsim_upload") or (
                SUBSCRIPTIES_PATH if os.path.exists(SUBSCRIPTIES_PATH) else None)
            if _drm_excel_src is not None:
                _drm_n_mi = _cached_aantal_klanten(
                    _file_mtime(_drm_excel_src if isinstance(_drm_excel_src, str) else ""),
                    upload=_drm_excel_src,
                )
        except Exception:
            pass
        if not _drm_n_mi.empty:
            st.caption(
                f"Adoptie-bewust: q(α={_drm_alpha:.1%}) = {_drm_q:.3f} — "
                f"Z_i(α) = M_i·q gebruikt als huidig niveau. "
                f"Drempel = extra abonnees boven Z_i(α) nodig voor S*+1."
            )

        _MAX_N_SEARCH = 100_000
        _drempel_rows = []

        for _, _row in _df_ov.iterrows():
            _code     = _row["Code"]
            _n_orig   = int(_row["n_klanten"])
            _lam_orig = float(_row["lambda_jr"])
            _lt_jr    = float(_row["LT_dagen"]) / 365

            # Adoptie-bewust: gebruik Z_i(α) = M_i·q(α) als huidig abonneeniveau
            _mi  = float(_drm_n_mi.get(str(_code), float(_n_orig))) if not _drm_n_mi.empty else float(_n_orig)
            _n   = int(round(_mi * _drm_q))
            _lam_pn = _lam_orig / _n_orig if _n_orig > 0 else _lam_orig
            _lam = _n * _lam_pn

            # S* op huidig adoptiepunt (Poisson-inverse)
            _s_now = (BPAOptimizationModel.inverse_service_level(_sl_d, _lam, _lt_jr)
                      if _n > 0 and _lam > 0 and _lt_jr > 0 else 0)

            if _n > 0 and _lam_pn > 0 and _lt_jr > 0:
                # Binary search: kleinste N_drempel waarbij S* > _s_now
                _lo, _hi = _n + 1, _n + _MAX_N_SEARCH
                _s_hi = BPAOptimizationModel.inverse_service_level(
                    _sl_d, _lam_pn * _hi, _lt_jr
                )
                if _s_hi <= _s_now:
                    _n_drempel = None
                else:
                    while _lo < _hi:
                        _mid = (_lo + _hi) // 2
                        _s_mid = BPAOptimizationModel.inverse_service_level(
                            _sl_d, _lam_pn * _mid, _lt_jr
                        )
                        if _s_mid > _s_now:
                            _hi = _mid
                        else:
                            _lo = _mid + 1
                    _n_drempel = _lo
            else:
                _n_drempel = None

            _extra = (_n_drempel - _n) if _n_drempel is not None else None
            _drempel_rows.append({
                "Code":          _code,
                "Omschrijving":  str(_row.get("Descr", ""))[:35],
                "Z huidig":      _n,
                "S* huidig":     _s_now,
                "Z voor S*+1":   _n_drempel if _n_drempel is not None else f">{_n + _MAX_N_SEARCH}",
                "Extra Z nodig": _extra,
                "λ/jr":          round(_lam, 4),
                "μ = λ·L":       round(_lam * _lt_jr, 4),
            })

        _tbl_d = pd.DataFrame(_drempel_rows).set_index("Code")
        _tbl_d_sorted = _tbl_d.sort_values("Extra Z nodig", na_position="last")
        # Styler.apply werkt niet met een niet-unieke index (dubbele 'Code').
        # Reset naar een unieke RangeIndex en verberg die in de weergave.
        if not _tbl_d_sorted.index.is_unique:
            _tbl_d_sorted = _tbl_d_sorted.reset_index()

        # Tabel weergeven met kleurcodering op basis van drempel
        def _kleur_drempel(row):
            v = row["Extra Z nodig"]
            if pd.isna(v):
                bg = "#d4edda"   # groen: geen drempel gevonden in zoekbereik
            elif int(v) <= 2:
                bg = "#f8d7da"   # rood: 1-2 extra subscripties
            elif int(v) <= 5:
                bg = "#fff3cd"   # oranje: 3-5 extra subscripties
            else:
                bg = "#d4edda"   # groen: 6+ extra subscripties
            return [f"background-color: {bg}"] * len(row)

        st.dataframe(
            _tbl_d_sorted.style
                .apply(_kleur_drempel, axis=1)
                .format({
                    "Z huidig":      "{:.0f}",
                    "S* huidig":     "{:.0f}",
                    "λ/jr":          "{:.4f}",
                    "μ = λ·L":       "{:.4f}",
                    "Extra Z nodig": lambda v: f"{int(v)}" if pd.notna(v) else "—",
                }),
            use_container_width=True,
            height=500,
        )

        # ── Bar chart: Extra N nodig per component ─────────────────────────
        _plot_d = _tbl_d_sorted[_tbl_d_sorted["Extra Z nodig"].notna()].copy()
        if not _plot_d.empty:
            import matplotlib.pyplot as _plt_d

            # Cap het aantal balken: bij honderden componenten wordt de grafiek
            # onleesbaar én PIL gooit een DecompressionBombError zodra het
            # gerenderde PNG > ~179 megapixels wordt. Toon de top-N met
            # de hoogste drempel (relevante "rode" gevallen eerst).
            _MAX_BARS = 60
            _n_total  = len(_plot_d)
            if _n_total > _MAX_BARS:
                _plot_d = _plot_d.nsmallest(_MAX_BARS, "Extra Z nodig")
                st.caption(
                    f"📊 Grafiek toont de **{_MAX_BARS}** componenten met de "
                    f"laagste drempel (van {_n_total} totaal). Volledige lijst "
                    f"staat in de tabel hierboven."
                )

            # Begrens figuur-breedte (max 32 inch) en zet expliciet dpi=100
            # om gegarandeerd onder de PIL-pixellimiet te blijven.
            _fig_w = min(max(8, len(_plot_d) * 0.55), 32)
            _fig_d, _ax_d = _plt_d.subplots(figsize=(_fig_w, 5), dpi=100)
            _ax_d.bar(
                range(len(_plot_d)),
                _plot_d["Extra Z nodig"].astype(int),
                color="#1976D2",
            )
            _ax_d.set_xticks(range(len(_plot_d)))
            _ax_d.set_xticklabels(
                _plot_d.index, rotation=45, ha="right", fontsize=9
            )
            _ax_d.set_ylabel("Extra subscriptions for S*+1", fontsize=11)
            _ax_d.set_title(
                f"Subscription threshold per component  (SL = {_sl_d:.1%})",
                fontsize=12,
            )
            _ax_d.grid(True, axis="y", alpha=0.3)
            _fig_d.tight_layout()
            st.pyplot(_fig_d)
            _plt_d.close(_fig_d)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 9 – CLASSIFICATIE
# ─────────────────────────────────────────────────────────────────────────────

with tab_classificatie:
    st.subheader("Classificatie — selectie voor BPA-beheer")
    st.caption(
        "Score alle artikelen uit de bron-Excel op prijs, klantlocaties en order-frequentie. "
        "Pas de gewichten en drempel aan; de selectie wordt — na 'Toepassen' — als whitelist "
        "doorgezet naar het tabblad 📊 Overzicht."
    )

    # ── Bron-Excel: vaste repo-Excel + vaste sheet ──────────────────────
    _cls_upload = None  # upload niet meer nodig; bestand staat in de repo
    _cls_bron = EXCEL_PATH
    _cls_sheet = "Filtered "  # vast — zelfde sheet als BPA-overzicht (MTBF(years) correct)
    st.caption(f"Bron-Excel: `{os.path.basename(EXCEL_PATH)}` — sheet: `{_cls_sheet.strip()}`")

    st.divider()

    # ── Parameters ──
    st.markdown("**Gewichten** _(worden automatisch genormaliseerd)_")
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        _w_prijs = st.slider("Gewicht prijs",    0.0, 1.0, 1/3, 0.05, key="cls_w_prijs")
    with _c2:
        _w_loc   = st.slider("Gewicht locaties", 0.0, 1.0, 1/3, 0.05, key="cls_w_loc")
    with _c3:
        _w_ord   = st.slider("Gewicht orders",   0.0, 1.0, 1/3, 0.05, key="cls_w_ord")

    # ── Selectiemethode: vast op "top X componenten" ────────────────────
    _sel_modus = "top_n"
    _thr = 0.0
    _top_pct = 20.0
    st.markdown("**Selectie: top X componenten (hoogste gewogen score)**")
    _top_n = st.number_input(
        "Aantal componenten (top X)", 1, 100_000, 100, 1, key="cls_top_n",
        help="De X componenten met de hoogste gewogen score (ná de harde "
             "filters) worden opgenomen in de lijst.",
    )

    st.markdown("**Niet-lineariteiten**")
    _ord_pow = st.slider("Orders-power", 1.0, 4.0, 2.0, 0.1, key="cls_ord_pow")

    st.markdown("**Min-filter drempels**")
    st.caption(
        "Artikelen onder deze drempels worden uitgesloten vóór de weging wordt toegepast."
    )
    _mf1, _mf2 = st.columns(2)
    with _mf1:
        _min_prijs = st.number_input(
            "Min. verkoopprijs (€)", 0.0, 100_000.0, 0.0, 10.0,
            key="cls_min_prijs",
            help="Artikelen met verkoopprijs < dit bedrag worden uitgesloten (harde filter).",
        )
    with _mf2:
        _min_orders = st.number_input(
            "Min. gem. orders/locatie", 0.0, 100.0, 0.0, 0.1,
            key="cls_min_orders",
            format="%.1f",
            help="Artikelen met gem. orders/locatie < deze waarde worden uitgesloten.",
        )

    # ── Aggregatiemethode: vast op geometrisch ──────────────────────────
    _score_methode = "geometrisch"
    st.markdown("**Aggregatiemethode: geometrisch** _(gewogen geometrisch gemiddelde — "
                "een zeer lage score op één dimensie trekt de totaalscore sterker omlaag)_")
    _epsilon = st.number_input(
        "ε (epsilon, verschuiving)", 0.001, 10.0, 1.0, 0.1,
        key="cls_epsilon",
        format="%.3f",
        help="Kleine constante waarmee elke score wordt verschoven voor de machtsverheffing "
             "(s̅ᴵ = (s + ε) / (100 + ε)). Standaard 1.0."
    )

    st.markdown("**Harde filters**")
    _min_loc = st.number_input("Min. klantlocaties", 0, 100, 5, 1, key="cls_min_loc")
    # ArticleType-filter: vast op critical + onbekend
    _art_types = ("critical", "onbekend")
    st.caption("ArticleType-filter (vast): `critical, onbekend`")

    _params = ClassificatieParams(
        threshold=float(_thr),
        selectie_modus=_sel_modus,
        top_n=int(_top_n),
        top_pct=float(_top_pct),
        weight_prijs=float(_w_prijs),
        weight_locaties=float(_w_loc),
        weight_orders=float(_w_ord),
        orders_power=float(_ord_pow),
        min_prijs=float(_min_prijs),
        min_orders=float(_min_orders),
        min_klantlocaties=int(_min_loc),
        article_type_filter=_art_types,
        score_methode=_score_methode,
        epsilon=float(_epsilon),
    )

    st.divider()

    # ── Run-knop ──
    _col_run, _col_apply = st.columns([1, 1])
    with _col_run:
        _run_cls = st.button("🔄 Bereken classificatie", type="primary", key="cls_run")
    with _col_apply:
        _apply_cls = st.button("✅ Toepassen op BPA-overzicht", key="cls_apply",
                               disabled=("cls_result" not in st.session_state))

    if _run_cls:
        try:
            with st.spinner("Classificatie berekenen…"):
                # De (trage) Excel-parse wordt gecachet, zodat alleen de
                # gevectoriseerde scoring opnieuw draait bij parameter-tweaks.
                if _cls_upload is not None:
                    _df_raw = _cached_laad_ruwe_dataset(0.0, _cls_sheet, _cls_upload)
                    _bron_excel = None
                else:
                    _df_raw = _cached_laad_ruwe_dataset(
                        _file_mtime(EXCEL_PATH), _cls_sheet
                    )
                    _bron_excel = str(EXCEL_PATH)
                _miss = controleer_kolommen(_df_raw)
                if _miss:
                    raise ValueError(f"Ontbrekende kolommen: {_miss}")
                # Eerst basis-filteren, daarna scoren: de min-max-normalisatie
                # gaat zo over de artikelenset NÁ de harde filters. Top-n volgt
                # op de gescoorde set.
                _df_basis    = pas_basis_filters_toe(_df_raw, _params)
                _df_scored   = bereken_scores(_df_basis, _params)
                _df_filtered = pas_topn_selectie_toe(_df_scored, _params)
                _payload     = bouw_selectie_payload(
                    _df_filtered, _params, bron_excel=_bron_excel
                )
            st.session_state.cls_result   = _df_filtered
            st.session_state.cls_payload  = _payload
            st.session_state.cls_params   = _params
            st.session_state.cls_raw      = _df_raw
            _sel_info = (
                f"top {_params.top_n}" if _params.selectie_modus == "top_n"
                else f"top {_params.top_pct:.0f}% per criterium" if _params.selectie_modus == "top_pct_all"
                else f"drempel ≥ {_params.threshold}"
            )
            st.toast(f"{_payload['n_items']} componenten geselecteerd "
                     f"({_sel_info})", icon="✅")
        except Exception as e:
            st.error(f"Fout tijdens classificatie: {e}")

    # ── Resultaten ──
    if "cls_result" in st.session_state:
        _res = st.session_state.cls_result
        _pl  = st.session_state.cls_payload

        _n_tot     = len(_res)
        _n_opnemen = (_res["Classificatie_Beslissing"] == "Opnemen in lijst").sum()

        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Na harde filters", _n_tot)
        _m2.metric("Opnemen in lijst", int(_n_opnemen),
                   delta=f"{_n_opnemen/_n_tot*100:.0f}%" if _n_tot else "—")
        _m3.metric("LT geupdate",  _pl["lt_overzicht"]["geupdate"])
        _m4.metric("LT default / ontbreekt",
                   _pl["lt_overzicht"]["default"] + _pl["lt_overzicht"]["ontbreekt"])

        # Tabel — sorteer op score, kleurcodering op beslissing
        _show_cols = [c for c in [
            "Verkooporderregel artikel.Artikel.Artikelcode", "Artikelcode", "Code",
            "ABC_categorie", "ArticleType",
            "Standaard verkoopprijs",
            "Aantal_klantlocaties_met_orders_5jr",
            "Gem_orders_per_klantlocatie_5jr",
            # MTBF: bron-kolom (originele waarde + eenheid) + genormaliseerd in jaren
            "MTBF(years)", "MTBF (years)", "MTBF_years",
            "MTBF(jaren)", "MTBF (jaren)",
            "MTBF (dagen)", "MTBF(dagen)",
            "MTBF (days)", "MTBF(days)", "MTBF_days",
            "MTBF",
            "MTBF_jaren",
            "Lambda_jr",
            "Score_Prijs", "Score_Locaties", "Score_Orders",
            "Gewogen_Score", "Classificatie_Beslissing",
            "Hoofdleverancier.Levertijd",
        ] if c in _res.columns]
        _df_show = _res[_show_cols].sort_values("Gewogen_Score", ascending=False)

        def _kleur_beslissing(v):
            return ("background-color: #c8e6c9" if v == "Opnemen in lijst"
                    else "background-color: #ffcdd2")

        st.dataframe(
            _df_show.style.map(_kleur_beslissing, subset=["Classificatie_Beslissing"]),
            use_container_width=True, height=500,
        )

        # Download
        _csv = _df_show.to_csv(sep=";", decimal=",", index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download gescoorde tabel (CSV)",
            data=_csv, file_name=f"classificatie_{date.today()}.csv",
            mime="text/csv",
        )

        # ── Drempel-sweep analyse ──
        if "cls_raw" in st.session_state:
            with st.expander("📉 Drempel-sweep: stabiliteit van de selectielijst", expanded=False):
                st.caption(
                    "Per filter wordt één drempel gevarieerd; de andere twee blijven op hun huidige waarde. "
                    "Jaccard-similariteit en Kendall\u2019s \u03c4 meten hoeveel de selectielijst "
                    "afwijkt t.o.v. de huidige instelling (referentie = rode stippellijn)."
                )
                _sw_raw    = st.session_state.cls_raw
                _sw_params = st.session_state.cls_params
                _sw_types  = set(s.lower() for s in _sw_params.article_type_filter)
                _sw_mask   = (
                    _sw_raw["ArticleType"].astype(str).str.strip().str.lower()
                    .isin(_sw_types)
                )
                _sw_scored = bereken_scores(_sw_raw[_sw_mask].copy(), _sw_params)

                _SW_LOC = "Aantal_klantlocaties_met_orders_5jr"
                _SW_PRI = "Standaard verkoopprijs"
                _SW_ORD = "Gem_orders_per_klantlocatie_5jr"

                def _sw_sel(ml, mp, mo):
                    """Hard filters + topn; returns frozenset of index labels."""
                    _t = _sw_scored.copy()
                    if ml > 0 and _SW_LOC in _t.columns:
                        _t = _t[_t[_SW_LOC].fillna(0) >= ml]
                    if mp > 0 and _SW_PRI in _t.columns:
                        _t = _t[_t[_SW_PRI].fillna(0) >= mp]
                    if mo > 0 and _SW_ORD in _t.columns:
                        _t = _t[_t[_SW_ORD].fillna(0) >= mo]
                    return frozenset(pas_topn_selectie_toe(_t, _sw_params).index)

                def _sweep_stats(col_dim, thr_vals, base_loc, base_pri, base_ord):
                    """Returns list of (thr, jaccard, kendall_tau) vs baseline."""
                    try:
                        from scipy.stats import kendalltau as _kt
                    except ImportError:
                        _kt = None
                    import numpy as _np_sw
                    _idx     = list(_sw_scored.index)
                    _base    = _sw_sel(base_loc, base_pri, base_ord)
                    _v_base  = _np_sw.array([1 if i in _base else 0 for i in _idx], dtype=_np_sw.int8)
                    out = []
                    for tv in thr_vals:
                        ml = tv if col_dim == _SW_LOC else base_loc
                        mp = tv if col_dim == _SW_PRI else base_pri
                        mo = tv if col_dim == _SW_ORD else base_ord
                        _new   = _sw_sel(ml, mp, mo)
                        _v_new = _np_sw.array([1 if i in _new else 0 for i in _idx], dtype=_np_sw.int8)
                        _inter = int((_v_base & _v_new).sum())
                        _union = int((_v_base | _v_new).sum())
                        _jac   = _inter / _union if _union > 0 else 1.0
                        if _kt is not None:
                            try:
                                _tau = float(_kt(_v_base, _v_new)[0])
                            except Exception:
                                _tau = float("nan")
                        else:
                            _tau = float("nan")
                        out.append((tv, _jac, _tau))
                    return out

                import matplotlib.pyplot as _plt_sw
                _sw_configs = [
                    (_SW_LOC, "Min. klantlocaties",      float(_sw_params.min_klantlocaties)),
                    (_SW_PRI, "Min. verkoopprijs (€)",   float(_sw_params.min_prijs)),
                    (_SW_ORD, "Min. orders/locatie",     float(_sw_params.min_orders)),
                ]
                _sw_fig, _sw_axes = _plt_sw.subplots(2, 3, figsize=(13, 5.5), sharex="col")
                for _ci, (_sw_col, _sw_lbl, _sw_cur) in enumerate(_sw_configs):
                    _ax_j = _sw_axes[0][_ci]
                    _ax_t = _sw_axes[1][_ci]
                    if _sw_col not in _sw_scored.columns:
                        for _ax in (_ax_j, _ax_t):
                            _ax.text(0.5, 0.5, "kolom niet gevonden", ha="center",
                                     va="center", transform=_ax.transAxes, fontsize=9)
                        continue
                    _sw_vals  = _sw_scored[_sw_col].fillna(0)
                    _sw_p95   = float(_sw_vals.quantile(0.95))
                    _sw_max   = max(_sw_p95, _sw_cur * 1.5, 1.0)
                    _sw_steps = [_sw_max * k / 39 for k in range(40)]
                    _sw_res   = _sweep_stats(
                        _sw_col, _sw_steps,
                        float(_sw_params.min_klantlocaties),
                        float(_sw_params.min_prijs),
                        float(_sw_params.min_orders),
                    )
                    _xs   = [r[0] for r in _sw_res]
                    _jacs = [r[1] for r in _sw_res]
                    _taus = [r[2] for r in _sw_res]
                    _ax_j.plot(_xs, _jacs, color="#1976d2", linewidth=2)
                    _ax_j.set_ylabel("Jaccard", fontsize=9)
                    _ax_j.set_ylim(-0.05, 1.05)
                    _ax_j.set_title(_sw_lbl, fontsize=9)
                    _ax_t.plot(_xs, _taus, color="#388e3c", linewidth=2)
                    _ax_t.set_ylabel("Kendall's \u03c4", fontsize=9)
                    _ax_t.set_ylim(-0.05, 1.05)
                    _ax_t.set_xlabel(_sw_lbl, fontsize=9)
                    if _sw_cur > 0:
                        for _ax in (_ax_j, _ax_t):
                            _ax.axvline(_sw_cur, color="#b71c1c", linestyle="--",
                                        linewidth=1.5, label=f"huidig: {_sw_cur:g}")
                            _ax.legend(fontsize=8)
                    for _ax in (_ax_j, _ax_t):
                        _ax.tick_params(labelsize=8)
                _plt_sw.tight_layout()
                st.pyplot(_sw_fig)
                _plt_sw.close(_sw_fig)

        # ── Verdeling: 3D-visualisatie + bar charts per criterium ──
        # Twee weergaven: (a) genormaliseerde scores (0–100/200) en
        # (b) de daadwerkelijke ruwe componentdata (€, #locaties, #orders).
        _score_cols = ["Score_Prijs", "Score_Locaties", "Score_Orders"]
        _raw_cols   = [
            "Standaard verkoopprijs",
            "Aantal_klantlocaties_met_orders_5jr",
            "Gem_orders_per_klantlocatie_5jr",
        ]
        _has_scores = all(c in _res.columns for c in _score_cols)
        _has_raw    = all(c in _res.columns for c in _raw_cols)

        if _has_scores or _has_raw:
            st.divider()
            st.markdown("### 📐 Verdeling per criterium")
            st.caption(
                "Visualiseer hoe de componenten zich verhouden op de drie criteria. "
                "De 3D-scatter toont de spreiding over álle criteria tegelijk; "
                "de histogrammen tonen per criterium hoe scheef (skewed) de verdeling is."
            )

            import matplotlib.pyplot as _plt_cls
            import matplotlib.ticker as _mt_cls
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registreert 3d-projectie)

            # Keuze databron: genormaliseerde scores vs ruwe componentdata.
            _bron_opties = []
            if _has_scores:
                _bron_opties.append("Genormaliseerde scores")
            if _has_raw:
                _bron_opties.append("Ruwe componentdata")
            _viz_bron = st.radio(
                "Databron voor visualisatie",
                _bron_opties,
                horizontal=True,
                key="cls_viz_bron",
            )

            if _viz_bron == "Ruwe componentdata":
                _viz_cols  = _raw_cols
                _crit_meta = {
                    _raw_cols[0]: ("Price (€)",      "#1976D2"),
                    _raw_cols[1]: ("Customer commonality",  "#388E3C"),
                    _raw_cols[2]: ("Orders / location", "#F57C00"),
                }
                _eenheid_x = "Value"
            else:
                _viz_cols  = _score_cols
                _crit_meta = {
                    "Score_Prijs":    ("Price",    "#1976D2"),
                    "Score_Locaties": ("Commonality", "#388E3C"),
                    "Score_Orders":   ("Orders",   "#F57C00"),
                }
                _eenheid_x = "Score"

            # Optionele log-schaal — handig bij sterk scheve ruwe data (prijs).
            _log_schaal = st.checkbox(
                "Log-schaal op assen (handig bij scheve ruwe data)",
                value=(_viz_bron == "Ruwe componentdata"),
                key="cls_viz_log",
            )

            _plot_df = _res.copy()
            for _c in _viz_cols:
                _plot_df[_c] = pd.to_numeric(_plot_df[_c], errors="coerce")
            _plot_df = _plot_df.dropna(subset=_viz_cols)
            if _log_schaal:
                # Log-schaal vereist strikt positieve waarden.
                _plot_df = _plot_df[(_plot_df[_viz_cols] > 0).all(axis=1)]

            if _plot_df.empty:
                st.info("Geen geldige data beschikbaar voor visualisatie "
                        "(controleer evt. de log-schaal-optie).")
            else:
                _cx, _cy, _cz = _viz_cols
                _lx, _ly, _lz = (_crit_meta[_cx][0], _crit_meta[_cy][0], _crit_meta[_cz][0])

                _opnemen_mask = (
                    _plot_df["Classificatie_Beslissing"] == "Opnemen in lijst"
                    if "Classificatie_Beslissing" in _plot_df.columns
                    else pd.Series(True, index=_plot_df.index)
                )

                # ── 3D-scatter: alle drie de criteria tegelijk ──
                _fig3d = _plt_cls.figure(figsize=(10, 8))
                _ax3d = _fig3d.add_subplot(111, projection="3d")

                for _mask, _kleur, _lbl in [
                    (_opnemen_mask,  "#2E7D32", "Include in list"),
                    (~_opnemen_mask, "#C62828", "Do not include"),
                ]:
                    _sub = _plot_df[_mask]
                    if not _sub.empty:
                        _ax3d.scatter(
                            _sub[_cx], _sub[_cy], _sub[_cz],
                            c=_kleur, label=_lbl, s=22, alpha=0.6,
                            edgecolors="none", depthshade=True,
                        )

                _ax3d.set_xlabel(_lx, fontsize=10, labelpad=12)
                _ax3d.set_ylabel(_ly, fontsize=10, labelpad=12)
                _ax3d.set_zlabel(_lz, fontsize=10, labelpad=12)
                if _log_schaal:
                    _ax3d.set_xscale("log")
                    _ax3d.set_yscale("log")
                    _ax3d.set_zscale("log")
                _ax3d.set_title(
                    f"3D distribution — {_viz_bron.lower()} ({len(_plot_df)} components)",
                    fontsize=12,
                )
                _ax3d.legend(fontsize=9, loc="upper left")
                _ax3d.view_init(elev=22, azim=-58)
                # tight_layout() knipt de z-as-label (orderfrequentie) van 3D-plots
                # weg. Zoom de 3D-box iets uit en gebruik ruime, gebalanceerde
                # marges zodat álle drie de assen + labels zichtbaar blijven
                # (de z-as 'Orders' staat bij deze kijkhoek aan de linkerkant).
                try:
                    _ax3d.set_box_aspect(None, zoom=0.82)
                except (TypeError, AttributeError):
                    pass  # oudere matplotlib zonder zoom-parameter
                _fig3d.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.06)
                st.pyplot(_fig3d)
                _plt_cls.close(_fig3d)

                # ── Bar charts (histogrammen) per criterium ──
                st.markdown("**Verdeling per criterium**")
                _hist_cols = st.columns(3)
                for _col_name, _slot in zip(_viz_cols, _hist_cols):
                    _vals = _plot_df[_col_name].astype(float)
                    _lbl, _kleur = _crit_meta[_col_name]
                    _figh, _axh = _plt_cls.subplots(figsize=(4.5, 3.2))
                    if _log_schaal and (_vals > 0).all():
                        _bins = np.logspace(
                            np.log10(_vals.min()), np.log10(_vals.max()), 20
                        )
                        _axh.set_xscale("log")
                    else:
                        _bins = 20
                    _axh.hist(
                        _vals, bins=_bins, color=_kleur,
                        edgecolor="white", alpha=0.85,
                    )
                    _mediaan = float(_vals.median())
                    _gem = float(_vals.mean())
                    _axh.axvline(_gem, color="#212121", linestyle="--",
                                 linewidth=1.2, label=f"Mean {_gem:,.1f}")
                    _axh.axvline(_mediaan, color="#757575", linestyle=":",
                                 linewidth=1.2, label=f"Median {_mediaan:,.1f}")
                    _axh.set_title(_lbl, fontsize=11)
                    _axh.set_xlabel(_eenheid_x, fontsize=9)
                    _axh.set_ylabel("Number of components", fontsize=9)
                    _axh.legend(fontsize=8)
                    _axh.grid(True, axis="y", alpha=0.3)
                    _figh.tight_layout()
                    with _slot:
                        st.pyplot(_figh)
                    _plt_cls.close(_figh)

                # ── Scheefheid (skewness) per criterium ──
                _skew_data = {
                    _crit_meta[c][0]: [
                        round(float(_plot_df[c].mean()), 2),
                        round(float(_plot_df[c].median()), 2),
                        round(float(_plot_df[c].skew()), 2),
                    ]
                    for c in _viz_cols
                }
                _skew_df = pd.DataFrame(
                    _skew_data, index=["Gemiddelde", "Mediaan", "Scheefheid"]
                ).T
                st.markdown("**Scheefheid (skewness) per criterium**")
                st.caption(
                    "Scheefheid > 0 = rechts-scheef (veel lage waarden, enkele uitschieters); "
                    "< 0 = links-scheef; ≈ 0 = symmetrisch."
                )
                st.dataframe(_skew_df, use_container_width=True)

    # ── Gewichten-sensitivity sweep ──────────────────────────────────────
    if "cls_result" in st.session_state:
        st.divider()
        st.markdown("### ⚖️ Gewichten-sensitivity")
        st.caption(
            "Varieer de drie criteria-gewichten (prijs / locaties / orders) over "
            "een simplex-raster en zie hoe stabiel de selectie is. Alle overige "
            "parameters (drempel/top-N, penalty's, harde filters) blijven gelijk. "
            "De huidige gewichten vormen de *baseline*."
        )

        _sw1, _sw2 = st.columns([1, 1])
        with _sw1:
            _sweep_step = st.select_slider(
                "Rasterresolutie (stap)",
                options=[0.5, 0.25, 0.2, 0.1, 0.05],
                value=0.1,
                key="cls_sweep_step",
                help="Kleiner = fijner raster en meer combinaties (langzamer). "
                     "0.1 ≈ 66 combinaties, 0.05 ≈ 231.",
            )
        with _sw2:
            _run_sweep = st.button("⚖️ Bereken gewichten-sweep", key="cls_run_sweep")

        if _run_sweep:
            st.session_state.cls_sweep_on = True

        if st.session_state.get("cls_sweep_on"):
            try:
                _params_json = json.dumps({
                    "threshold":               _params.threshold,
                    "selectie_modus":          _params.selectie_modus,
                    "top_n":                   _params.top_n,
                    "weight_prijs":            _params.weight_prijs,
                    "weight_locaties":         _params.weight_locaties,
                    "weight_orders":           _params.weight_orders,
                    "orders_power":            _params.orders_power,
                    "min_prijs":               _params.min_prijs,
                    "min_orders":              _params.min_orders,
                    "min_klantlocaties":       _params.min_klantlocaties,
                    "article_type_filter":     list(_params.article_type_filter),
                    "score_methode":           _params.score_methode,
                    "epsilon":                 _params.epsilon,
                }, sort_keys=True)
                _per_artikel, _per_combo = _cached_weight_sweep(
                    st.session_state.cls_result, _params_json, float(_sweep_step),
                    versie=2,
                )
            except Exception as e:
                st.error(f"Fout tijdens gewichten-sweep: {e}")
                _per_artikel = _per_combo = None

            if _per_artikel is not None and not _per_artikel.empty:
                _n_combos = len(_per_combo)
                _altijd = int((_per_artikel["Stabiliteit"] == "altijd").sum())
                _soms   = int((_per_artikel["Stabiliteit"] == "soms").sum())
                _nooit  = int((_per_artikel["Stabiliteit"] == "nooit").sum())
                _base_n = int(_per_artikel["In_baseline"].sum())

                _sm1, _sm2, _sm3, _sm4, _sm5 = st.columns(5)
                _sm1.metric("Combinaties", _n_combos)
                _sm2.metric("In baseline", _base_n)
                _sm3.metric("Altijd geselecteerd", _altijd)
                _sm4.metric("Soms (gevoelig)", _soms)
                _sm5.metric("Gem. selectiegrootte",
                            f"{_per_combo['n_opnemen'].mean():.0f}")

                import matplotlib.pyplot as _plt_sw

                _cc1, _cc2 = st.columns(2)

                # (a) Simplex-scatter: kleur = aantal opnemen per combinatie
                with _cc1:
                    st.markdown("**Selectiegrootte per gewicht-combinatie**")
                    _fig1, _ax1 = _plt_sw.subplots(figsize=(5, 4))
                    _sc = _ax1.scatter(
                        _per_combo["weight_prijs"], _per_combo["weight_orders"],
                        c=_per_combo["n_opnemen"], cmap="viridis", s=90,
                        edgecolor="white", linewidth=0.5,
                    )
                    _ax1.set_xlabel("price weight")
                    _ax1.set_ylabel("order weight")
                    _ax1.set_title("# included (commonality = remainder)", fontsize=10)
                    _ax1.grid(True, alpha=0.3)
                    _fig1.colorbar(_sc, ax=_ax1, label="# included")
                    _fig1.tight_layout()
                    st.pyplot(_fig1)
                    _plt_sw.close(_fig1)
                    st.caption("gewicht locaties = 1 − prijs − orders.")

                # (b) Stabiliteitsverdeling
                with _cc2:
                    st.markdown("**Stabiliteit van artikelen**")
                    _fig2, _ax2 = _plt_sw.subplots(figsize=(5, 4))
                    _ax2.bar(
                        ["always", "sometimes", "never"],
                        [_altijd, _soms, _nooit],
                        color=["#2ca02c", "#ff7f0e", "#d62728"],
                        edgecolor="white",
                    )
                    for _i, _v in enumerate([_altijd, _soms, _nooit]):
                        _ax2.text(_i, _v, str(_v), ha="center", va="bottom",
                                  fontsize=9)
                    _ax2.set_ylabel("number of components")
                    _ax2.set_title("'sometimes' = selection depends on weighting",
                                   fontsize=10)
                    _ax2.grid(True, axis="y", alpha=0.3)
                    _fig2.tight_layout()
                    st.pyplot(_fig2)
                    _plt_sw.close(_fig2)

                # (c) Histogram van selectie-frequentie
                st.markdown("**Verdeling van de selectie-frequentie**")
                _fig3, _ax3 = _plt_sw.subplots(figsize=(9, 2.6))
                _ax3.hist(
                    _per_artikel["Selectie_frequentie"], bins=20,
                    range=(0, 1), color="#1f77b4", edgecolor="white", alpha=0.85,
                )
                _ax3.set_xlabel("fraction of combinations selected")
                _ax3.set_ylabel("number of components")
                _ax3.grid(True, axis="y", alpha=0.3)
                _fig3.tight_layout()
                st.pyplot(_fig3)
                _plt_sw.close(_fig3)

                # ── Rangorde-robuustheid ──────────────────────────────────
                if "spearman" in _per_combo.columns:
                    st.markdown("#### 🔢 Effect op de volgorde (rangorde)")
                    st.caption(
                        "Naast wél/niet in de set (Jaccard) telt ook de volgorde. "
                        "Spearman/Kendall = rangcorrelatie over de hele kandidaat-set "
                        "(1 = identieke volgorde). RBO weegt de **top** zwaar — "
                        "relevant voor top-N-prioritering. 'Rank-shift' = aantal "
                        "posities dat een artikel opschuift t.o.v. de baseline."
                    )

                    _cm = _per_combo.copy()
                    _rk1, _rk2, _rk3, _rk4 = st.columns(4)
                    _rk1.metric("Laagste Spearman",
                                f"{_cm['spearman'].min():.2f}",
                                help="Worst-case rangcorrelatie over alle wegingen.")
                    _rk2.metric("Laagste RBO (top-zwaar)",
                                f"{_cm['rbo'].min():.2f}")
                    _rk3.metric("Grootste rank-shift",
                                f"{int(_cm['max_rank_shift'].max())}")
                    _rk4.metric("Gem. rank-shift",
                                f"{_cm['mean_rank_shift'].mean():.1f}")

                    # Per dominant criterium: welk criterium verstoort de
                    # volgorde het meest als het zwaarder weegt?
                    _crit_naam = {0: "price", 1: "commonality", 2: "orders"}
                    _wcols = ["weight_prijs", "weight_locaties", "weight_orders"]
                    _cm["_dominant"] = (
                        _cm[_wcols].values.argmax(axis=1)
                    )
                    _cm["Dominant criterium"] = _cm["_dominant"].map(_crit_naam)
                    _grp = (
                        _cm.groupby("Dominant criterium")[
                            ["spearman", "kendall", "rbo", "mean_rank_shift"]
                        ].mean().reindex(["prijs", "locaties", "orders"])
                    )

                    _rc1, _rc2 = st.columns(2)
                    with _rc1:
                        st.markdown("**Rang-stabiliteit per dominant criterium**")
                        _fig4, _ax4 = _plt_sw.subplots(figsize=(5, 4))
                        _xpos = np.arange(len(_grp))
                        _bw = 0.4
                        _ax4.bar(_xpos - _bw/2, _grp["spearman"], _bw,
                                 label="Spearman", color="#1f77b4")
                        _ax4.bar(_xpos + _bw/2, _grp["rbo"], _bw,
                                 label="RBO (top)", color="#ff7f0e")
                        _ax4.set_xticks(_xpos)
                        _ax4.set_xticklabels(_grp.index)
                        _ax4.set_ylim(0, 1)
                        _ax4.set_ylabel("avg. correlation vs. baseline")
                        _ax4.set_title("Higher = ranking more stable",
                                       fontsize=10)
                        _ax4.legend(fontsize=8)
                        _ax4.grid(True, axis="y", alpha=0.3)
                        _fig4.tight_layout()
                        st.pyplot(_fig4)
                        _plt_sw.close(_fig4)
                        st.caption("Het criterium met de **laagste** balken "
                                   "verstoort de volgorde het sterkst.")

                    with _rc2:
                        st.markdown("**Gem. rank-shift per dominant criterium**")
                        _fig5, _ax5 = _plt_sw.subplots(figsize=(5, 4))
                        _ax5.bar(_grp.index, _grp["mean_rank_shift"],
                                 color=["#2ca02c", "#9467bd", "#d62728"],
                                 edgecolor="white")
                        for _i, _v in enumerate(_grp["mean_rank_shift"]):
                            _ax5.text(_i, _v, f"{_v:.0f}", ha="center",
                                      va="bottom", fontsize=9)
                        _ax5.set_ylabel("avg. rank shift")
                        _ax5.set_title("Higher = ranking shifts more",
                                       fontsize=10)
                        _ax5.grid(True, axis="y", alpha=0.3)
                        _fig5.tight_layout()
                        st.pyplot(_fig5)
                        _plt_sw.close(_fig5)
                        st.caption("Grotere verschuiving = minder robuuste "
                                   "prioritering.")

                    _worst = _grp["spearman"].idxmin()
                    _best = _grp["spearman"].idxmax()
                    st.info(
                        f"➡️ De volgorde is het **gevoeligst** voor het gewicht van "
                        f"**{_worst}** (laagste rangcorrelatie) en het **robuustst** "
                        f"voor **{_best}**. Onderbouw het gewicht van *{_worst}* het "
                        f"zorgvuldigst; daar bepaalt je keuze de prioritering."
                    )

                # (d) Resultatentabel per artikel
                st.markdown("**Resultaten per artikel** "
                            "_(gesorteerd op selectie-frequentie)_")
                _only_soms = st.checkbox(
                    "Toon alleen gevoelige artikelen (Stabiliteit = 'soms')",
                    value=False, key="cls_sweep_only_soms",
                )
                _tabel = (_per_artikel[_per_artikel["Stabiliteit"] == "soms"]
                          if _only_soms else _per_artikel)
                st.dataframe(
                    _tabel, use_container_width=True, height=420, hide_index=True,
                    column_config={
                        "Selectie_frequentie": st.column_config.ProgressColumn(
                            "Selectie_frequentie", min_value=0.0, max_value=1.0,
                            format="%.2f",
                        ),
                    },
                )

                with st.expander("Detail per gewicht-combinatie"):
                    st.dataframe(_per_combo, use_container_width=True,
                                 hide_index=True)

                _sweep_csv = _per_artikel.to_csv(
                    sep=";", decimal=",", index=False
                ).encode("utf-8")
                st.download_button(
                    "⬇️ Download sweep-resultaten (CSV)",
                    data=_sweep_csv,
                    file_name=f"gewichten_sweep_{date.today()}.csv",
                    mime="text/csv",
                    key="cls_sweep_dl",
                )

    # ── Apply: schrijf bpa_selectie.json + invalideer overzicht ──
    if _apply_cls and "cls_payload" in st.session_state:
        try:
            schrijf_selectie_json(st.session_state.cls_payload, SELECTIE_PATH)
            st.session_state.pop("overzicht_df", None)
            st.success(
                f"✅ Selectie opgeslagen in {SELECTIE_PATH}. "
                f"Open tab 📊 Overzicht — de basisvoorraden worden opnieuw berekend "
                f"met **{st.session_state.cls_payload['n_items']}** componenten als whitelist."
            )
        except Exception as e:
            st.error(f"Kon bpa_selectie.json niet schrijven: {e}")

    # ── Verwijder bestaande selectie (alle artikelen weer actief) ──
    st.divider()
    if os.path.exists(SELECTIE_PATH):
        if st.button("🗑️ Verwijder huidige classificatie-selectie (BPA gebruikt weer alle Excel-codes)"):
            try:
                os.remove(SELECTIE_PATH)
                st.session_state.pop("overzicht_df", None)
                st.toast("Selectie verwijderd — BPA gebruikt weer de standaard Excel-filters.", icon="🗑️")
                st.rerun()
            except Exception as e:
                st.error(f"Kon bestand niet verwijderen: {e}")
    else:
        st.info("Geen actieve classificatie-selectie. De BPA-tool gebruikt momenteel de standaard Excel-filters.")
