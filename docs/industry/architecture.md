# AI in Architecture & Construction

AI is reshaping how buildings are conceived, engineered, and built — compressing months of design iteration into hours, predicting structural failures before ground breaks, and optimizing energy performance across the building lifecycle. From generative floorplan synthesis to autonomous site monitoring, the industry is undergoing its deepest transformation since CAD replaced the drafting table.

> **By the numbers (2025):** AI in construction market projected at $8.6B by 2027. Firms using generative design report 40-70% faster schematic design cycles. BIM + AI reduces RFI (Request for Information) costs by up to 30%.

---

## Overview

| Domain | Key AI Techniques | Typical Stack |
|---|---|---|
| Generative Design | Evolutionary algorithms, GANs, diffusion models | Grasshopper, Hypar, Finch |
| BIM Intelligence | NLP on IFC data, anomaly detection, clash prediction | ifcopenshell, LangChain, Revit API |
| Site Analysis | Computer vision, satellite imagery, GIS ML | QGIS + ML, Google Earth Engine |
| Energy Simulation | Surrogate ML models, Gaussian processes | EnergyPlus + scikit-learn, ladybug |
| Cost Estimation | Regression, XGBoost, NLP on specs | ProEst, custom XGBoost pipelines |
| Structural Analysis | FEM + ML surrogates, topology optimization | OpenSeesPy, PyTorch |
| Visualization | Diffusion models, NeRF, real-time rendering | Stable Diffusion, Veras, Midjourney |

---

## Key AI Use Cases

### Generative Design & Concept Generation

Generative design inverts the traditional workflow: instead of drawing a solution, you define **constraints** (area targets, circulation requirements, structural spans, daylight factors) and let algorithms explore millions of configurations simultaneously.

Modern pipelines combine:

- **Evolutionary / genetic algorithms** — explore large combinatorial spaces (room adjacency graphs, massing options)
- **Diffusion models** — generate architectural imagery from text or sketch prompts
- **Reinforcement learning** — optimize floorplan layouts against multi-objective reward functions (daylight, circulation efficiency, structural regularity)

**Key capability:** Generate 500+ massing variations overnight; filter by solar access, GFA, and cost — reducing schematic phase from 6 weeks to 3 days.

---

### Building Information Modeling (BIM) + AI

BIM models contain rich semantic geometry (walls, slabs, MEP routes, structural members) stored in IFC format. AI unlocks three capabilities traditional BIM cannot:

1. **Natural language queries over model data** — "Show me all structural columns with fire rating below 2 hours"
2. **Automated clash detection** — ML classifiers prioritize which clashes are truly critical vs. acceptable
3. **Change impact prediction** — given a design change, predict downstream coordination issues before they become RFIs

---

### Site Analysis & Urban Planning

Computer vision pipelines process satellite/drone imagery to extract:

- **Topography & slope analysis** — identify buildable areas, drainage paths
- **Existing vegetation & tree canopy** — automated survey for planning submissions
- **Solar potential maps** — irradiance modeling from LiDAR + weather data
- **Traffic & pedestrian flow** — video analytics to inform building entrances and parking ratios

Tools like **Delve** (Sidewalk Labs/Alphabet) optimize entire urban blocks for density, sunlight, and walkability simultaneously.

---

### Energy Performance Simulation

Full EnergyPlus simulations take 20-60 minutes per model. ML surrogate models trained on thousands of simulations reduce this to milliseconds, enabling:

- Real-time energy feedback during schematic design
- Parametric optimization over envelope, glazing ratio, and HVAC type
- Portfolio-scale energy benchmarking across hundreds of buildings

---

### Construction Cost Estimation

Historically done by quantity surveyors using manual takeoffs, AI now:

- Extracts quantities directly from BIM/IFC models
- Applies NLP to read specification documents and map to cost databases (RSMeans, Uniformat)
- Predicts cost overrun risk from project parameters (scope complexity, contract type, site conditions)

XGBoost models trained on historical project data routinely achieve ±8-12% accuracy vs. ±20-30% for manual estimates at the schematic stage.

---

### Client Presentation & Visualization

Diffusion models have transformed architectural visualization:

- **Concept imagery** — Midjourney/Stable Diffusion generates photorealistic renders from rough sketches in seconds
- **Style transfer** — apply material and lighting variations to existing renders
- **Veras (Enscape)** — real-time AI rendering directly inside Revit/SketchUp
- **NeRF / Gaussian Splatting** — generate walk-through 3D scenes from drone photos of the site

What previously required a 3D visualization studio and 2 weeks now takes an afternoon.

---

### Structural Analysis

AI accelerates structural engineering through:

- **Topology optimization** — minimize material use while meeting load requirements (used in complex facade nodes, transfer structures)
- **Surrogate models** — replace expensive FEM solvers for early-stage parametric exploration
- **Damage detection** — CV models on drone footage identify concrete cracking, spalling, and corrosion

---

## Top AI Tools & Platforms

| Tool | Provider | Primary Use Case | Integration |
|---|---|---|---|
| **Autodesk Forma** | Autodesk | Massing, daylight, wind analysis | Cloud, Revit plugin |
| **Spacemaker** | Autodesk (acquired) | Urban-scale generative design | Web, API |
| **TestFit** | TestFit Inc. | Rapid site yield / feasibility | Web, REST API |
| **Hypar** | Hypar Inc. | Generative BIM functions (code-first) | Web, C# SDK |
| **Arkio** | Arkio | VR collaborative design + AI | HMD, desktop |
| **Veras** | Enscape / Chaos | AI rendering inside Revit/SketchUp | Revit, SketchUp plugin |
| **Finch** | Finch3D | Floorplan layout optimization | Revit plugin, web |
| **Delve** | Sidewalk Labs (Google) | Urban planning optimization | Web (enterprise) |
| **Midjourney** | Midjourney Inc. | Concept visualization, mood boards | Discord, API |
| **Stable Diffusion** | Stability AI | Customizable architectural imagery | Local, API, ComfyUI |
| **SketchUp + AI plugins** | Trimble + ecosystem | Early massing, material suggestion | Extension Warehouse |
| **Rhino + Grasshopper + ML** | McNeel + Shapediver | Parametric ML-driven design | Python/GH scripting |
| **Cove.tool** | Cove.tool Inc. | Energy + carbon optimization | Revit, web |
| **OpenSeesPy + ML** | Open source | Structural analysis surrogates | Python |

---

## Technology Stack

### Generative Design with Python — Parametric Floorplan Generation

This example uses a constraint-satisfaction approach to generate and score floorplan layouts from a spatial program.

```python
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Room:
    name: str
    min_area: float   # m²
    max_area: float
    adjacencies: List[str] = field(default_factory=list)  # desired neighbours

@dataclass
class PlacedRoom:
    name: str
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def overlaps(self, other: "PlacedRoom", gap: float = 0.5) -> bool:
        return not (
            self.x + self.w + gap <= other.x or
            other.x + other.w + gap <= self.x or
            self.y + self.h + gap <= other.y or
            other.y + other.h + gap <= self.y
        )

def adjacency_score(layout: List[PlacedRoom], program: List[Room]) -> float:
    """Reward rooms that should be adjacent for being close together."""
    room_map = {r.name: r for r in layout}
    score = 0.0
    for spec in program:
        if spec.name not in room_map:
            continue
        cx, cy = room_map[spec.name].centroid
        for adj_name in spec.adjacencies:
            if adj_name not in room_map:
                continue
            ax, ay = room_map[adj_name].centroid
            dist = math.hypot(cx - ax, cy - ay)
            score -= dist  # penalise distance between required neighbours
    return score

def area_penalty(layout: List[PlacedRoom], program: List[Room]) -> float:
    """Penalise rooms outside their area band."""
    spec_map = {s.name: s for s in program}
    penalty = 0.0
    for room in layout:
        spec = spec_map.get(room.name)
        if spec is None:
            continue
        if room.area < spec.min_area:
            penalty -= (spec.min_area - room.area) * 10
        elif room.area > spec.max_area:
            penalty -= (room.area - spec.max_area) * 5
    return penalty

def generate_layout(
    program: List[Room],
    site_width: float = 20.0,
    site_depth: float = 15.0,
    population: int = 200,
    generations: int = 500,
) -> List[PlacedRoom]:
    """Simple evolutionary layout generator."""

    def random_individual() -> List[PlacedRoom]:
        placed = []
        for spec in program:
            w = random.uniform(math.sqrt(spec.min_area) * 0.7, math.sqrt(spec.max_area) * 1.3)
            h = random.uniform(spec.min_area / w, spec.max_area / w)
            w = min(w, site_width)
            h = min(h, site_depth)
            x = random.uniform(0, max(0, site_width - w))
            y = random.uniform(0, max(0, site_depth - h))
            placed.append(PlacedRoom(spec.name, x, y, w, h))
        return placed

    def fitness(ind: List[PlacedRoom]) -> float:
        overlap_penalty = sum(
            -50.0
            for i, a in enumerate(ind)
            for b in ind[i + 1:]
            if a.overlaps(b)
        )
        return adjacency_score(ind, program) + area_penalty(ind, program) + overlap_penalty

    def mutate(ind: List[PlacedRoom]) -> List[PlacedRoom]:
        ind = [PlacedRoom(r.name, r.x, r.y, r.w, r.h) for r in ind]
        idx = random.randrange(len(ind))
        r = ind[idx]
        r.x = max(0, min(site_width - r.w,  r.x + random.gauss(0, 0.8)))
        r.y = max(0, min(site_depth - r.h, r.y + random.gauss(0, 0.8)))
        return ind

    # Initialise population
    pop = [random_individual() for _ in range(population)]

    for gen in range(generations):
        pop.sort(key=fitness, reverse=True)
        survivors = pop[: population // 2]
        offspring = [mutate(random.choice(survivors)) for _ in range(population // 2)]
        pop = survivors + offspring

    best = max(pop, key=fitness)
    return best


# --- Example usage ---
program = [
    Room("living",   20, 35, adjacencies=["dining", "terrace"]),
    Room("dining",   15, 25, adjacencies=["living", "kitchen"]),
    Room("kitchen",  12, 20, adjacencies=["dining"]),
    Room("master",   18, 30, adjacencies=["ensuite"]),
    Room("ensuite",   6, 12, adjacencies=["master"]),
    Room("bedroom2", 12, 20, adjacencies=[]),
    Room("terrace",  10, 30, adjacencies=["living"]),
]

layout = generate_layout(program, site_width=22, site_depth=16)
for room in layout:
    print(f"{room.name:12s}  {room.area:5.1f} m²  @ ({room.x:.1f}, {room.y:.1f})")
```

---

### Image-to-3D Concept — Stable Diffusion for Architectural Visualization

```python
import base64
import httpx
from pathlib import Path

STABILITY_API_KEY = "sk-..."   # set via env var in production
API_URL = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

ARCHITECTURAL_STYLE_SUFFIX = (
    ", architectural visualization, photorealistic exterior render, "
    "golden hour lighting, 8k, sharp focus, shot on Hasselblad"
)

def generate_concept_render(
    prompt: str,
    output_path: str = "concept_render.png",
    negative_prompt: str = "cartoon, sketch, hand-drawn, watermark, blurry",
    aspect_ratio: str = "16:9",
) -> Path:
    """
    Generate a photorealistic architectural concept image via Stable Diffusion 3.
    Returns path to saved PNG.
    """
    full_prompt = prompt + ARCHITECTURAL_STYLE_SUFFIX

    response = httpx.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "image/*",
        },
        data={
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "model": "sd3-large",
            "cfg_scale": 7.5,
        },
        timeout=60,
    )
    response.raise_for_status()

    out = Path(output_path)
    out.write_bytes(response.content)
    print(f"Saved render to {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def sketch_to_render(
    sketch_path: str,
    prompt: str,
    strength: float = 0.65,   # how much to deviate from input sketch
    output_path: str = "render_from_sketch.png",
) -> Path:
    """Image-to-image: turn a rough sketch into a photorealistic render."""
    img_data = Path(sketch_path).read_bytes()

    response = httpx.post(
        "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "image/*",
        },
        data={
            "prompt": prompt + ARCHITECTURAL_STYLE_SUFFIX,
            "image": base64.b64encode(img_data).decode(),
            "strength": strength,
            "model": "sd3-large",
            "output_format": "png",
        },
        timeout=90,
    )
    response.raise_for_status()
    out = Path(output_path)
    out.write_bytes(response.content)
    return out


# --- Example usage ---
generate_concept_render(
    prompt=(
        "contemporary courtyard house, exposed concrete and timber, "
        "floor-to-ceiling glazing, surrounded by mature eucalyptus trees, "
        "Melbourne suburb"
    )
)
```

---

### BIM Data Analysis — Parsing IFC Files with ifcopenshell + LLM Queries

```python
import ifcopenshell
import ifcopenshell.util.element as ifc_util
import os
import json
from anthropic import Anthropic

client = Anthropic()

def extract_model_summary(ifc_path: str) -> dict:
    """Extract key statistics and element data from an IFC model."""
    model = ifcopenshell.open(ifc_path)

    summary = {
        "schema": model.schema,
        "project": None,
        "element_counts": {},
        "materials": [],
        "spaces": [],
        "properties": {},
    }

    # Project info
    projects = model.by_type("IfcProject")
    if projects:
        p = projects[0]
        summary["project"] = {"name": p.Name, "description": p.Description}

    # Element counts
    for ifc_type in [
        "IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcDoor",
        "IfcWindow", "IfcStair", "IfcRoof", "IfcSpace",
    ]:
        elements = model.by_type(ifc_type)
        if elements:
            summary["element_counts"][ifc_type] = len(elements)

    # Spaces (rooms) with area
    for space in model.by_type("IfcSpace"):
        props = ifc_util.get_psets(space)
        area = None
        for pset in props.values():
            area = pset.get("NetFloorArea") or pset.get("GrossFloorArea") or area
        summary["spaces"].append({
            "name": space.Name or space.LongName,
            "area_m2": round(float(area), 2) if area else None,
        })

    # Materials
    for mat in model.by_type("IfcMaterial"):
        if mat.Name not in summary["materials"]:
            summary["materials"].append(mat.Name)

    return summary


def query_bim_model(ifc_path: str, user_question: str) -> str:
    """
    Answer natural-language questions about a BIM model using Claude.
    Extracts model data first, then lets the LLM reason over it.
    """
    print(f"Parsing IFC model: {ifc_path}")
    summary = extract_model_summary(ifc_path)

    system_prompt = """You are an expert BIM analyst. You receive structured data extracted
from an IFC (Industry Foundation Classes) building model and answer questions
about it. Be precise with numbers. If data is missing, say so clearly.
Always suggest what additional IFC data would help answer the question fully."""

    user_message = f"""BIM Model Data:
{json.dumps(summary, indent=2)}

Question: {user_question}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def check_clash_risk(ifc_path: str) -> list[dict]:
    """
    Basic programmatic clash detection: flag structural elements
    with no fire rating property set.
    """
    model = ifcopenshell.open(ifc_path)
    issues = []

    structural_types = ["IfcColumn", "IfcBeam", "IfcSlab"]
    for ifc_type in structural_types:
        for element in model.by_type(ifc_type):
            psets = ifc_util.get_psets(element)
            fire_rating = None
            for pset in psets.values():
                fire_rating = pset.get("FireRating") or fire_rating
            if not fire_rating:
                issues.append({
                    "type": ifc_type,
                    "id": element.id(),
                    "name": element.Name,
                    "issue": "Missing FireRating property",
                })

    return issues


# --- Example usage ---
# answer = query_bim_model("project.ifc", "What is the total number of rooms and their average area?")
# print(answer)

# issues = check_clash_risk("project.ifc")
# print(f"Found {len(issues)} elements missing fire rating data")
```

---

### Energy Optimization — ML Model Predicting Building Energy Consumption

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import joblib

# --- Feature schema (maps to EnergyPlus simulation outputs) ---
FEATURE_COLUMNS = [
    "floor_area_m2",
    "aspect_ratio",           # long side / short side
    "glazing_ratio_n",        # north facade glazing %
    "glazing_ratio_s",        # south facade glazing %
    "glazing_ratio_e",
    "glazing_ratio_w",
    "wall_u_value",           # W/m²K
    "roof_u_value",
    "slab_u_value",
    "glazing_u_value",
    "shgc",                   # solar heat gain coefficient
    "infiltration_ach",       # air changes per hour
    "hvac_cop",               # HVAC coefficient of performance
    "climate_zone",           # 1-8 ASHRAE climate zones
    "orientation_deg",        # building long-axis rotation from north
    "occupancy_density",      # m² per person
    "lighting_power_w_m2",
    "equipment_power_w_m2",
    "num_floors",
]

TARGET = "eui_kwh_m2_yr"   # Energy Use Intensity


def generate_synthetic_training_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    In practice, run EnergyPlus via OpenStudio for each sample.
    Here we simulate realistic relationships for demonstration.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "floor_area_m2":       rng.uniform(200, 5000, n_samples),
        "aspect_ratio":        rng.uniform(1.0, 4.0, n_samples),
        "glazing_ratio_n":     rng.uniform(0.10, 0.80, n_samples),
        "glazing_ratio_s":     rng.uniform(0.10, 0.80, n_samples),
        "glazing_ratio_e":     rng.uniform(0.10, 0.60, n_samples),
        "glazing_ratio_w":     rng.uniform(0.10, 0.60, n_samples),
        "wall_u_value":        rng.uniform(0.15, 1.80, n_samples),
        "roof_u_value":        rng.uniform(0.10, 0.50, n_samples),
        "slab_u_value":        rng.uniform(0.10, 0.80, n_samples),
        "glazing_u_value":     rng.uniform(0.80, 3.50, n_samples),
        "shgc":                rng.uniform(0.20, 0.70, n_samples),
        "infiltration_ach":    rng.uniform(0.10, 1.50, n_samples),
        "hvac_cop":            rng.uniform(2.5, 6.0, n_samples),
        "climate_zone":        rng.integers(1, 9, n_samples),
        "orientation_deg":     rng.uniform(0, 90, n_samples),
        "occupancy_density":   rng.uniform(8, 25, n_samples),
        "lighting_power_w_m2": rng.uniform(5, 20, n_samples),
        "equipment_power_w_m2":rng.uniform(8, 30, n_samples),
        "num_floors":          rng.integers(1, 20, n_samples),
    })

    # Synthetic EUI formula (physics-informed approximation)
    df[TARGET] = (
        80
        + (df["wall_u_value"] - 0.5) * 30
        + (df["glazing_u_value"] - 2.0) * 25
        + df["glazing_ratio_s"] * df["shgc"] * -40   # south solar = passive gain benefit
        + df["glazing_ratio_n"] * 15                  # north glazing = heat loss
        + df["infiltration_ach"] * 20
        - (df["hvac_cop"] - 3.5) * 10
        + (df["climate_zone"] - 4) * 8
        + df["lighting_power_w_m2"] * 1.5
        + df["equipment_power_w_m2"] * 1.2
        + rng.normal(0, 5, n_samples)               # measurement noise
    ).clip(30, 400)

    return df


def train_energy_surrogate(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")

    print(f"Test MAPE : {mape:.2f}%")
    print(f"CV R²     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipe


def optimize_glazing(
    model: Pipeline,
    base_params: dict,
    climate_zone: int = 4,
) -> pd.DataFrame:
    """
    Grid search over glazing ratios to find the optimal configuration
    for a given building profile.
    """
    results = []
    for gn in np.arange(0.10, 0.81, 0.10):      # north glazing
        for gs in np.arange(0.10, 0.81, 0.10):   # south glazing
            params = {**base_params, "glazing_ratio_n": gn, "glazing_ratio_s": gs}
            eui = model.predict(pd.DataFrame([params]))[0]
            results.append({"glazing_n": gn, "glazing_s": gs, "eui": round(eui, 1)})

    df = pd.DataFrame(results).sort_values("eui")
    return df


# --- Train and save ---
training_data = generate_synthetic_training_data(5000)
surrogate = train_energy_surrogate(training_data)
joblib.dump(surrogate, "energy_surrogate.pkl")

# --- Optimize a sample building ---
base_building = {
    "floor_area_m2": 1200,
    "aspect_ratio": 2.5,
    "glazing_ratio_n": 0.30,
    "glazing_ratio_s": 0.50,
    "glazing_ratio_e": 0.20,
    "glazing_ratio_w": 0.20,
    "wall_u_value": 0.35,
    "roof_u_value": 0.20,
    "slab_u_value": 0.30,
    "glazing_u_value": 1.40,
    "shgc": 0.38,
    "infiltration_ach": 0.30,
    "hvac_cop": 4.5,
    "climate_zone": 4,
    "orientation_deg": 0,
    "occupancy_density": 12,
    "lighting_power_w_m2": 9,
    "equipment_power_w_m2": 15,
    "num_floors": 4,
}

optimal = optimize_glazing(surrogate, base_building)
print("\nTop 5 glazing configurations by EUI:")
print(optimal.head())
```

---

## Best Workflow

AI does not replace the design process — it accelerates and augments each phase. Here is the optimal human-AI workflow for a commercial project in 2025:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  AI-AUGMENTED DESIGN WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────┘

 1. CLIENT BRIEF
    └── AI: LLM extracts requirements, flags ambiguities, checks against
            zoning regulations, generates design brief summary doc
            Tool: Custom brief analyzer (see section below)

         │
         ▼

 2. SITE ANALYSIS
    └── AI: Satellite image analysis (slope, vegetation, solar access)
            Shadow study automation, pedestrian flow modelling
            Tool: Autodesk Forma, Google Earth Engine + ML

         │
         ▼

 3. CONCEPT GENERATION
    └── AI: Diffusion models produce massing variations from text prompts
            Generative layout tools explore 100s of schemes overnight
            Tool: Midjourney, Stable Diffusion, Spacemaker, Finch

         │
         ▼

 4. SCHEMATIC DESIGN
    └── AI: Real-time energy feedback per design move
            BIM auto-population from sketch geometry
            Automated area schedule vs. brief compliance check
            Tool: Forma, Cove.tool, Hypar, Veras

         │
         ▼

 5. DESIGN DEVELOPMENT
    └── AI: Structural layout optimisation (load paths, column grids)
            MEP clash prediction before coordination drawings issued
            Spec document drafting from BIM element properties
            Tool: Revit + ML plugins, TestFit, Claude API

         │
         ▼

 6. DOCUMENTATION
    └── AI: BIM-to-spec automated drafting
            NLP review of specs against project requirements
            Automated code compliance cross-check
            Tool: LLM APIs, ifcopenshell, Procore AI

         │
         ▼

 7. CONSTRUCTION
    └── AI: Site monitoring (drone CV for progress tracking)
            Schedule risk prediction from daily site reports
            Quality inspection (crack/defect detection via CV)
            Cost-to-complete forecasting
            Tool: OpenSpace, Procore AI, custom CV pipelines

─────────────────────────────────────────────────────────────────────

AI TOUCHPOINTS SUMMARY

  High-value, time-saving AI interventions:
  ★★★  Concept generation (days → hours)
  ★★★  Energy optimisation (weeks → seconds with surrogate models)
  ★★☆  BIM clash triage (reduces RFI volume by ~30%)
  ★★☆  Cost estimation (±8% vs. ±25% manual at schematic stage)
  ★☆☆  Site monitoring (still requires human judgement on actions)
```

---

## Building a Design Assistant

An LLM-powered architectural brief analyzer that extracts requirements, suggests design approaches, and cross-checks zoning compliance.

```python
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional
from anthropic import Anthropic

client = Anthropic()

@dataclass
class DesignBrief:
    project_type: str
    site_area_m2: Optional[float]
    gross_floor_area_m2: Optional[float]
    num_floors: Optional[int]
    key_spaces: list[str]
    budget_range: Optional[str]
    sustainability_targets: list[str]
    planning_constraints: list[str]
    client_aspirations: list[str]
    ambiguities: list[str]   # questions that need clarification


EXTRACTION_SYSTEM = """You are a senior architect with 20 years of experience.
Given a client brief in natural language, extract and structure the following fields
as valid JSON. If a field cannot be determined, use null. Be precise with numbers.

Return ONLY valid JSON, no markdown fences, no explanation.

Schema:
{
  "project_type": "string (e.g. 'residential', 'commercial office', 'mixed-use')",
  "site_area_m2": number or null,
  "gross_floor_area_m2": number or null,
  "num_floors": integer or null,
  "key_spaces": ["list of required spaces/rooms"],
  "budget_range": "string or null",
  "sustainability_targets": ["list of targets e.g. 'Passivhaus', '6 Star NatHERS'"],
  "planning_constraints": ["list of known constraints"],
  "client_aspirations": ["qualitative statements about character and feel"],
  "ambiguities": ["list of questions the architect should ask to resolve gaps"]
}"""

DESIGN_ADVISOR_SYSTEM = """You are a principal architect giving design strategy advice.
Given a structured project brief, suggest:
1. THREE distinct design approaches (each 2-3 sentences)
2. Key structural system options and why
3. Sustainability strategy recommendations
4. Three risks or challenges to flag early
5. Recommended consultant team

Be specific, practical, and reference real precedents where helpful."""

ZONING_CHECKER_SYSTEM = """You are an experienced town planner. Given project parameters
and a planning zone description, identify:
1. Likely compliance issues (with specific concerns)
2. Items that require council discretion or variation
3. Information needed before a formal DA/permit can be lodged
4. Recommended pre-application meeting agenda items

Note: This is preliminary desk-based advice only, not formal planning advice."""


def extract_brief(raw_text: str) -> DesignBrief:
    """Parse a freeform client brief into a structured DesignBrief."""
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": raw_text}],
    )
    data = json.loads(response.content[0].text)
    return DesignBrief(**data)


def suggest_design_approaches(brief: DesignBrief) -> str:
    """Generate design strategy suggestions from a structured brief."""
    brief_json = json.dumps(asdict(brief), indent=2)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=DESIGN_ADVISOR_SYSTEM,
        messages=[{"role": "user", "content": f"Project Brief:\n{brief_json}"}],
    )
    return response.content[0].text


def check_zoning_compliance(brief: DesignBrief, zone_description: str) -> str:
    """Cross-check brief against a planning zone description."""
    message = f"""Project Brief:
{json.dumps(asdict(brief), indent=2)}

Planning Zone Description:
{zone_description}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=ZONING_CHECKER_SYSTEM,
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text


def run_design_assistant(raw_brief: str, zone_description: str) -> dict:
    """Full pipeline: extract → advise → compliance check."""
    print("Step 1: Extracting brief requirements...")
    brief = extract_brief(raw_brief)

    print("Step 2: Generating design approaches...")
    design_advice = suggest_design_approaches(brief)

    print("Step 3: Checking zoning compliance...")
    compliance_notes = check_zoning_compliance(brief, zone_description)

    if brief.ambiguities:
        print(f"\n⚠  {len(brief.ambiguities)} ambiguities identified — follow up with client:")
        for i, q in enumerate(brief.ambiguities, 1):
            print(f"   {i}. {q}")

    return {
        "brief": asdict(brief),
        "design_advice": design_advice,
        "compliance_notes": compliance_notes,
    }


# --- Example usage ---
sample_brief = """
We need a new office headquarters for our 80-person software company in Melbourne's
inner north. We own the site at 45 Smith St — it's about 1,200 sqm. We want something
that really reflects our culture: open, collaborative, lots of natural light. We'd love
a rooftop terrace if we can get it. We're aiming for around $8-12M construction budget
and we definitely want to hit a 5-star Green Star rating. We want to be in by late 2027.
The building needs a mix of open-plan workstations, 8-10 meeting rooms, a large
all-hands space for 120 people, a café, and basement parking for at least 20 cars.
We're flexible on height but don't want to go over 6 storeys.
"""

sample_zone = """
Zone: Commercial 1 Zone (C1Z) — City of Yarra
Preferred uses: Office, retail, food & drink premises, shop
Maximum building height: 11 metres without permit, discretionary above
Car parking: Maximum 1 space per 100m² GFA (activity centre)
Heritage overlay: No
Design & Development Overlay: DDO9 applies — specific urban design guidelines
Active ground floor requirement: Yes, minimum 80% of ground floor frontage
"""

result = run_design_assistant(sample_brief, sample_zone)
print("\n--- STRUCTURED BRIEF ---")
print(json.dumps(result["brief"], indent=2))
print("\n--- DESIGN ADVICE ---")
print(result["design_advice"])
print("\n--- COMPLIANCE NOTES ---")
print(result["compliance_notes"])
```

---

## ROI & Metrics

| Metric | Before AI | With AI | Source / Method |
|---|---|---|---|
| Schematic design duration | 4-8 weeks | 1-2 weeks | Generative design tools (Spacemaker internal data) |
| Massing options reviewed | 5-20 | 500-5,000 | Automated generation |
| Cost estimate accuracy (schematic) | ±20-30% | ±8-12% | XGBoost trained on historical data |
| Energy model turnaround | 2-5 days | Minutes | Surrogate ML model |
| Visualization production | 2-3 weeks | 1-2 days | Diffusion model + Veras |
| BIM RFI volume | Baseline | -25-35% | Clash pre-detection |
| Site safety incident prediction | Reactive | 72h advance warning | CV + historical incident ML |
| Quantity takeoff time | 2-3 days | 2-4 hours | BIM-to-estimate automation |

### Calculating Design Iteration ROI

```python
def calculate_design_iteration_roi(
    hourly_rate: float,           # principal architect hourly rate
    manual_iterations: int,       # iterations in traditional process
    ai_iterations: int,           # iterations possible with AI
    hours_per_iteration_manual: float,
    hours_per_iteration_ai: float,
    project_fee: float,           # total design fee
) -> dict:
    manual_cost   = manual_iterations * hours_per_iteration_manual * hourly_rate
    ai_cost       = ai_iterations * hours_per_iteration_ai * hourly_rate
    time_saved_h  = (manual_iterations * hours_per_iteration_manual) - \
                    (ai_iterations * hours_per_iteration_ai)
    cost_saved    = manual_cost - ai_cost
    fee_pct_saved = cost_saved / project_fee * 100

    return {
        "manual_design_cost":   f"${manual_cost:,.0f}",
        "ai_design_cost":       f"${ai_cost:,.0f}",
        "cost_saved":           f"${cost_saved:,.0f}",
        "hours_freed":          f"{time_saved_h:.0f} hours",
        "fee_margin_improvement": f"{fee_pct_saved:.1f}%",
        "iterations_increase":  f"{ai_iterations / manual_iterations:.1f}x",
    }


print(calculate_design_iteration_roi(
    hourly_rate=180,
    manual_iterations=8,
    ai_iterations=60,
    hours_per_iteration_manual=12,
    hours_per_iteration_ai=0.5,
    project_fee=250_000,
))
# {'manual_design_cost': '$17,280', 'ai_design_cost': '$5,400',
#  'cost_saved': '$11,880', 'hours_freed': '66 hours',
#  'fee_margin_improvement': '4.8%', 'iterations_increase': '7.5x'}
```

---

## Compliance & Risks

### Building Code Compliance

AI-generated designs are still subject to the full suite of statutory requirements. AI tools do not replace the architect's professional duty of care.

| Risk Area | Issue | Mitigation |
|---|---|---|
| NCC / IBC compliance | AI may generate non-compliant room dimensions, egress paths | Human QA checklist at each milestone |
| Fire engineering | Automated layouts may miss compartmentation requirements | Fire engineer review before DD |
| Accessibility (DDA / ADA) | Generative tools rarely optimise for access compliance by default | Add accessibility constraints to generative model objectives |
| Structural adequacy | Surrogate ML models are interpolators — extrapolation is dangerous | Structural engineer sign-off on all members |
| Energy certificates | ML energy predictions are indicative only | Full EnergyPlus / IES-VE run for NCC Section J / NABERS |

### Copyright of AI-Generated Designs

This is an evolving legal area (2025 status):

- **Australia:** Copyright Act 1968 does not currently protect purely AI-generated works. The human architect's creative input (prompts, selection, curation) may be protectable.
- **EU:** AI Act (2024) requires disclosure of AI-generated content in regulated contexts. Building permit submissions may require disclosure.
- **US:** Copyright Office position: human authorship required. AI-assisted works with "sufficient human creative control" may qualify.
- **Practical rule:** Document your creative decisions, iterations, and modifications to AI outputs. The selection, curation, and modification process demonstrates human authorship.

### Accuracy Verification Checklist

```
Before submitting any AI-assisted work to a client or authority:

□ Verify generated areas against brief (±5% tolerance)
□ Check all AI cost estimates against current RSMeans / Rawlinson's rates
□ Confirm energy model assumptions match actual climate data
□ Structural geometry reviewed by registered structural engineer
□ AI-generated specification clauses checked against project standards
□ Copyright and attribution noted for all AI-generated imagery
□ IFC model clash check run with current discipline models
□ Planning compliance independently verified by town planner
```

---

## Tips & Tricks

| Tip | Category | Why It Matters |
|---|---|---|
| Use ControlNet with Stable Diffusion for sketch-to-render | Visualization | Preserves your spatial intent while adding photorealism |
| Constrain generative tools with structural grid first | Design | Unconstrained generation produces unbuildable layouts |
| Train surrogate energy models on YOUR climate data | Energy | Generic models drift when applied to unusual climates |
| ifcopenshell is free and handles IFC2x3, IFC4, IFC4.3 | BIM | No vendor lock-in; scriptable in any Python environment |
| Always specify units in LLM prompts (m², not "area") | LLM | Ambiguous units cause errors in extracted structured data |
| Use Hypar for code-first generative BIM (C#) | BIM/Generative | Version-controllable, reproducible, team-shareable |
| Prompt Midjourney with real precedents ("like BIG's Bjarke Ingels") | Visualization | Named references dramatically improve stylistic coherence |
| Surrogate models: validate against 20% holdout EnergyPlus runs | Energy | Prevents overfitting to training simulation set |
| Store all AI design decisions in a decision log | Compliance | Demonstrates professional judgement and human authorship |
| Run AI cost estimates at multiple design stages for trend analysis | Cost | Trend matters more than point estimate at early stages |
| Use Grasshopper + Wallacei for multi-objective evolutionary design | Generative | Pareto-front visualisation helps client understand trade-offs |
| Pre-application meetings: bring AI-generated options to show range | Planning | Demonstrates thorough site analysis to planners |

---

## Related Topics

- [AI in Computer Vision](../computer-vision/index.md) — Object detection and segmentation for site monitoring
- [Generative AI](../generative-ai/index.md) — Diffusion models and GANs underlying architectural image tools
- [LLM Agents](../llm/agents.md) — Agentic workflows for multi-step design automation
- [RAG Systems](../llm/rag.md) — Retrieval-augmented generation for building codes and standards
- [Optimization](../optimization/index.md) — Genetic algorithms and multi-objective optimization for generative design
- [Multi-Objective Optimization](../optimization/multi-objective.md) — Pareto-front methods for balancing energy, cost, and area
- [MLOps](../mlops/index.md) — Deploying surrogate energy models to production
- [AI in Industry](index.md) — Overview of AI across all industries
