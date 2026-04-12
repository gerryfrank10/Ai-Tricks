# AI in Agriculture

AI is reshaping how farmers grow food, manage land, and navigate markets — moving agriculture from intuition-driven decisions made seasonally to data-driven precision actions made daily. From satellites monitoring crop stress across thousands of acres to smartphone apps that diagnose plant disease from a single photo, the tools available to modern agri-businesses have fundamentally changed what is operationally possible.

> **By the numbers (2025):** The global AI in agriculture market is valued at $1.7B and projected to reach $4.7B by 2028 (CAGR ~20%). Precision agriculture adopters report average yield improvements of 10–20%. AI-guided variable-rate irrigation reduces water usage by 20–50% on instrumented farms. According to the FAO, feeding a projected 9.7 billion people by 2050 requires a 70% increase in food production — AI is a core enabler of that target.

---

## Overview

Agriculture sits at the intersection of biology, chemistry, meteorology, logistics, and economics — all domains where AI delivers measurable value. The key drivers of adoption are:

- **Labour shortages** pushing automation into harvesting, monitoring, and spraying
- **Climate volatility** increasing the need for responsive, sensor-driven irrigation and pest management
- **Commodity price pressure** forcing tighter margins and requiring more precise input management
- **Data abundance** from cheap IoT sensors, CubeSats, and consumer drones providing more agronomic data than any human can manually analyse

The result is a stack of interoperable tools — satellite imagery platforms, IoT soil networks, machine learning yield forecasters, and LLM-powered advisory systems — that together enable what the industry calls **precision agriculture**: doing the right thing, in the right place, at the right time.

| Domain | Key AI Techniques | Leading Platforms |
|---|---|---|
| Crop Monitoring | Computer vision, NDVI analysis | Taranis, Climate FieldView, OneSoil |
| Soil & Irrigation | IoT + ML, reinforcement learning | CropX, Arable, Farmers Edge |
| Pest & Disease | CNN classification, edge inference | Plantix, Blue River Technology |
| Yield Prediction | Gradient boosting, LSTMs | Ag-Analytics, Climate FieldView |
| Supply Chain | Time-series forecasting, NLP | IBM EIS, Granular |
| Livestock | CV, wearable IoT, anomaly detection | Connecterra, Allflex |

---

## Key AI Use Cases

### Precision Farming & Crop Monitoring

Precision farming replaces uniform field management with **zone-level decisions** calibrated to actual soil variability, crop growth stage, and microclimate. AI enables this by continuously processing satellite, drone, and ground sensor data to produce actionable prescription maps — files that instruct variable-rate applicators to apply exactly the right amount of fertiliser, seed, or herbicide at each GPS coordinate.

Key capabilities:
- **NDVI and NDRE time-series** track crop vigour week-over-week, flagging underperforming zones before yield loss becomes irreversible
- **In-season nitrogen management** — platforms like Climate FieldView combine satellite greenness with historical yield maps to generate top-dress N recommendations at the sub-field level
- **Prescription map generation** — exported to John Deere Operations Center, Trimble Ag, or CNH AFS for direct machine integration

---

### Drone & Satellite Imagery Analysis

Drone and satellite imagery form the visual backbone of modern precision agriculture. Two complementary workflows exist:

**Satellite (medium-resolution, high-frequency):**
Platforms like Planet Labs, Sentinel-2 (free, ESA), and Maxar provide imagery at 3–10 m resolution on revisit cycles of 1–5 days. Agromonitoring API and OneSoil aggregate these feeds and expose processed vegetation indices via API or web dashboard — no data science skills required.

**Drone (high-resolution, on-demand):**
Farm-operated multispectral drones (DJI Agras, Parrot Sequoia) fly 50–200 metre altitudes to capture imagery at 2–5 cm resolution. Platforms like Taranis use flight imagery for canopy-level pest and disease scouting, while DroneDeploy and Pix4Dfields handle stitching, NDVI rendering, and prescription export.

The standard analytical workflow:

```
Raw imagery → Stitching/orthorectification → Index computation (NDVI/NDRE/CWSI)
    → Anomaly zone detection → Agronomist review → Prescription map → Machine upload
```

---

### Soil Health & Irrigation Optimization

Soil is the most heterogeneous input in farming — pH, organic matter, texture, and moisture can vary dramatically within a single field. AI-driven soil platforms address this in two ways:

**Soil sensing networks (IoT):**
Sensor nodes from CropX, Sentek, and Arable are buried at multiple depths to measure volumetric water content, temperature, and electrical conductivity in real time. ML models running in the cloud translate raw sensor streams into **irrigation triggers** and **nutrient availability estimates**, pushing alerts to a mobile app when action is required.

**Variable-rate prescription:**
When combined with EC mapping and soil sampling results, ML models (typically gradient boosted trees trained on historical yield and soil data) generate soil management zones and zone-specific seeding and fertiliser rates. Farmers Edge and Trimble Ag both offer this as a service — a field technician visits, installs sensors, and the platform handles all modelling.

---

### Pest & Disease Detection

Early, accurate identification of crop pathogens and pest pressure is one of the highest-ROI applications of computer vision in agriculture. The challenge is speed: a disease that covers 5% of a crop canopy one week may devastate 60% the next.

Current approaches:
- **Smartphone apps (Plantix by PEAT):** A farmer photographs a symptomatic leaf; a CNN trained on 500,000+ annotated images returns a top-3 diagnosis with confidence score and treatment recommendation within seconds. Available in 18 languages. Used by 10+ million farmers globally.
- **In-field cameras (Taranis):** High-resolution cameras mounted on equipment or deployed on poles capture canopy-level imagery continuously. Deep learning models detect insects, fungal lesions, and nutrient deficiencies at densities too low for the human eye to spot reliably.
- **Autonomous spraying (See & Spray by John Deere / Blue River Technology):** Computer vision on a spray boom identifies individual weeds and triggers nozzles only when a weed is detected — reducing herbicide use by up to 77% vs. broadcast spraying.

---

### Yield Prediction & Harvest Planning

Accurate yield forecasts 4–8 weeks before harvest enable better decisions across the entire value chain: contract pricing, logistics scheduling, storage allocation, and labour hiring.

Inputs used by modern yield prediction models:
- Historical yield maps (from yield monitors on combines)
- In-season satellite NDVI time-series
- Soil type and management zone data
- Weather (temperature, rainfall, GDDs accumulated)
- Planting date and hybrid/variety data

Platforms like **Ag-Analytics** and **Climate FieldView** expose these forecasts directly in their dashboards. USDA's **NASS** publishes aggregate county-level yield forecasts, and tools like Ag-Analytics expose these via API for integration into trading or procurement systems.

---

### Supply Chain & Market Price Forecasting

Price volatility is a fundamental risk for farmers and agri-businesses. AI augments traditional commodity market analysis by processing non-traditional signals:

- **Satellite vegetation indices at scale** — services like Orbital Insight monitor crop conditions globally to predict harvest volumes before official government estimates
- **Weather and climate forecasting** — IBM Environmental Intelligence Suite ingests NOAA, ECMWF, and proprietary weather model output to produce crop-specific risk scores
- **Demand signal integration** — platforms like Granular (Corteva) link farm production data with commodity prices, helping farmers decide when to sell and which contracts to accept

---

### Livestock Monitoring & Health

AI is transforming livestock management from reactive to predictive, particularly in dairy, poultry, and swine operations.

Key applications:
- **Estrus and health detection (Connecterra, Allflex SCR):** Ear tags and leg-worn accelerometers track movement, rumination, and feeding behaviour. ML algorithms detect deviations from normal patterns 24–48 hours before visible clinical signs, enabling early intervention
- **Lameness scoring (Cainthus/IAG):** Computer vision cameras in dairy sheds analyse gait automatically, flagging lame animals for hoof-trimmer inspection
- **Poultry house monitoring:** Computer vision systems count birds, detect mortality clusters, and monitor average weight gain — replacing daily manual counts
- **Automated weighing and sorting:** Load cells + computer vision sort animals by weight class for targeted feeding without manual handling

---

## Top AI Tools & Platforms

| Tool | Provider | Category | Key Feature | Free Tier? | Website |
|---|---|---|---|---|---|
| Operations Center | John Deere | Farm Management | Machine data integration, field records, prescription maps | Yes (basic) | operations.deere.com |
| Climate FieldView | Bayer | Precision Agronomy | Satellite imagery, yield maps, N-management | Freemium | climate.com |
| Granular | Corteva | Farm Business Mgmt | Profit mapping, agronomic planning, benchmarking | No | granular.ag |
| Taranis | Taranis (Indigo) | Aerial Scouting | Sub-cm drone imagery, AI pest/disease detection | No | taranis.ag |
| Arable Mark | Arable | IoT / Microclimate | In-field weather + canopy sensor, ET estimation | No | arable.com |
| CropX | CropX | Soil & Irrigation | Wireless soil sensors, AI irrigation scheduling | No | cropx.com |
| Farmers Edge | Farmers Edge | Full-Stack Precision | Weather stations, satellite, soil, advisory | No | farmersedge.ca |
| See & Spray / Blue River | John Deere | Autonomous Spraying | Computer vision herbicide targeting, 77% reduction | No | bluerivertechnology.com |
| Plantix | PEAT GmbH | Disease Detection | CNN plant disease ID from smartphone photo | Yes (free) | plantix.net |
| IBM EIS | IBM | Weather & Climate Risk | Hyperlocal weather, crop risk scoring, supply chain | No | ibm.com/environmental-intelligence |
| Ag-Analytics | Ag-Analytics | Analytics & API | Yield forecasts, USDA data API, field analytics | Freemium | analytics.ag |
| Agromonitoring API | Agromonitoring | Satellite API | NDVI, EVI, NRI via REST API for any polygon | Freemium | agromonitoring.com |
| Trimble Ag | Trimble | Precision Guidance | GPS guidance, field data management, agronomic tools | No | trimble.com/agriculture |
| AFS Connect | CNH Industrial | Fleet & Field Mgmt | Case IH / New Holland machine integration | No | caseih.com/afsconnect |
| Prospera | Valmont Industries | Computer Vision | In-field camera AI for scouting and canopy analysis | No | prospera.ag |
| OneSoil | OneSoil | Satellite Analytics | Free NDVI maps, zone delineation, field history | Yes (free) | onesoil.ai |
| DroneDeploy | DroneDeploy | Drone Mapping | Drone flight planning, stitching, NDVI, prescription | Freemium | dronedeploy.com |

---

## Technology Behind the Tools

### Computer Vision for Crop & Pest Detection

Computer vision is the most visually intuitive AI application in agriculture — a model "looks" at an image and classifies what it sees.

**How it works (plain English):**

1. A large dataset of annotated images is assembled (e.g., 500,000 leaf photos labelled "healthy", "late blight", "aphid damage")
2. A convolutional neural network (CNN) — most commonly a ResNet or EfficientNet architecture — is trained to extract visual features (texture, colour patterns, edge shapes) associated with each class
3. The trained model is deployed either in the cloud (user uploads a photo) or on edge hardware (mounted camera processes in real time)
4. The model outputs a class label plus a confidence score; below a threshold, the system flags for human agronomist review

**Which tools use it:**

| Tool | CV Application | Where inference runs |
|---|---|---|
| Plantix | Leaf disease classification from smartphone camera | Cloud (mobile upload) |
| Taranis | Canopy pest/disease from drone imagery | Cloud |
| See & Spray (Blue River) | Weed vs. crop classification on spray boom | Edge (real-time, on-machine) |
| Prospera | In-field scouting camera — pest counts | Edge + cloud |
| Cainthus/IAG | Dairy cow gait scoring | Edge (barn cameras) |

---

### Satellite / Drone NDVI Analysis Workflow

NDVI (Normalized Difference Vegetation Index) measures the ratio of near-infrared (NIR) to red light reflected by a canopy. Healthy, dense vegetation absorbs red light and reflects NIR strongly; stressed or sparse crops do the opposite.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NDVI ANALYSIS WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Satellite/Drone         Processing Platform    Farm Dashboard  │
│  ─────────────           ──────────────────     ─────────────── │
│                                                                 │
│  Sentinel-2/Planet  ──►  Agromonitoring API ──► OneSoil map    │
│  DJI Multispectral  ──►  DroneDeploy/Pix4D  ──► Prescription   │
│  Parrot Sequoia     ──►  Taranis platform   ──► Scout alerts   │
│                                                                 │
│  Steps:                                                         │
│  1. Raw bands (Red, NIR) downloaded                            │
│  2. Atmospheric correction applied                             │
│  3. NDVI = (NIR - Red) / (NIR + Red) computed per pixel       │
│  4. Temporal differencing flags anomaly zones                  │
│  5. Zones exported as shapefiles or prescription maps          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Values range from -1 to +1. Healthy crops typically score 0.6–0.9; stressed crops fall below 0.4; bare soil is near 0.1.

---

### IoT Sensor Networks for Soil & Weather

An instrumented farm deploys a mesh of wireless nodes — typically LoRaWAN or cellular — that transmit readings every 15–60 minutes to a cloud platform.

```
Sensor Layer               Connectivity          Cloud Platform
────────────               ────────────          ──────────────
Soil moisture (TDR)   ─┐
Soil temperature      ─┤  LoRaWAN / 4G  ──►   CropX / Arable
Soil EC               ─┤                        Farmers Edge
Canopy temperature    ─┤                        Cloud ML engine
Air temp/RH/wind      ─┤                           │
Rain gauge            ─┘                           ▼
                                           Irrigation trigger
                                           Nutrient alert
                                           Mobile notification
```

The cloud ML engine compares live readings against crop water demand models (Penman-Monteith ET equations calibrated with local weather) to determine whether to trigger an irrigation zone. No agronomic expertise is required from the farmer — the platform handles the decision logic.

---

### ML for Yield Forecasting

Yield forecasting blends remote sensing, weather, and agronomic data into a structured prediction problem.

| Input Category | Example Variables | Source |
|---|---|---|
| Soil | Organic matter %, texture, pH, historical productivity | Soil surveys, field sampling |
| Weather | Growing Degree Days, total rainfall, heat stress events | NOAA, IBM EIS, Arable |
| Crop phenology | Planting date, hybrid maturity rating, canopy NDVI | FieldView, Taranis |
| Historical yields | Prior 5–10 years of yield monitor data | John Deere Ops Center |
| Management | Seeding rate, fertiliser applied, irrigation events | Granular, farm records |

**Model types used:** Gradient Boosting (XGBoost/LightGBM) for structured tabular data; LSTMs and Transformers for time-series NDVI trajectories. Ensemble methods combining both are common in research.

**Output:** Field-level yield estimate (bu/acre or t/ha) with confidence interval, updated weekly as the season progresses.

---

### LLM-Powered Farm Advisory Chatbots

Large language models are entering agriculture as the "ask a question, get an answer" interface layer on top of sensor and agronomic data. Examples:

- **John Deere's** integration of GPT-4 into Operations Center allows farmers to query their field data in plain English: "Which fields are below 70% field capacity?" or "Summarise nitrogen applications this season"
- **Climate FieldView** has piloted conversational agronomic advice drawing on weather forecasts and in-season data
- **Farmers Edge** offers SMS and app-based advisory that synthesises sensor data with agronomic knowledge bases

The general architecture:

```
Farmer question (natural language)
        │
        ▼
   LLM (GPT-4 / Claude)
        │  ◄── Farm data context (field records, sensor readings,
        │       weather forecast, crop model outputs)
        ▼
Grounded agronomic recommendation
        │
        ▼
Mobile app / SMS delivery
```

The key challenge is **hallucination prevention** — ensuring the LLM cites actual sensor readings and agronomic guidelines rather than generating plausible-sounding but incorrect advice. Production systems gate all outputs through agronomist review workflows or restrict the LLM to retrieval-augmented generation (RAG) over verified knowledge bases.

---

## Best Workflow: The Precision Farming Cycle

The following diagram shows a complete season-long precision farming workflow with specific platform assignments at each stage.

```
╔══════════════════════════════════════════════════════════════════════════╗
║               FULL-SEASON PRECISION FARMING WORKFLOW                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. PRE-SEASON SOIL SAMPLING                                             ║
║     ┌─────────────────────────────────────────────────────┐              ║
║     │ Tools: Farmers Edge EC mapping + CropX sensors       │              ║
║     │ Outputs: Soil management zones, variable-rate maps   │              ║
║     │ Platform: Upload to John Deere Operations Center     │              ║
║     └────────────────────────┬────────────────────────────┘              ║
║                              │                                           ║
║  2. PLANTING                 ▼                                           ║
║     ┌─────────────────────────────────────────────────────┐              ║
║     │ Tools: Trimble guidance + Ops Center prescription    │              ║
║     │ Outputs: Variable seeding rate applied per zone      │              ║
║     │ Data logged: Planting date, hybrid, population       │              ║
║     └────────────────────────┬────────────────────────────┘              ║
║                              │                                           ║
║  3. EARLY GROWTH MONITORING  ▼                                           ║
║     ┌─────────────────────────────────────────────────────┐              ║
║     │ Tools: OneSoil (free) → Taranis (drone scouting)    │              ║
║     │ Satellite: Sentinel-2 via Agromonitoring API         │              ║
║     │ Alerts: Emergence gaps, early pest/weed pressure     │              ║
║     └────────────────────────┬────────────────────────────┘              ║
║                              │                                           ║
║  4. IRRIGATION & PEST MGMT   ▼                                           ║
║     ┌─────────────────────────────────────────────────────┐              ║
║     │ Irrigation: CropX soil sensors → automated triggers  │              ║
║     │ Pest detection: Plantix (field scouting) + Taranis   │              ║
║     │ Spraying: See & Spray (targeted herbicide, -77% use) │              ║
║     │ Weather alerts: Arable microclimate + IBM EIS risk   │              ║
║     └────────────────────────┬────────────────────────────┘              ║
║                              │                                           ║
║  5. YIELD PREDICTION         ▼                                           ║
║     ┌─────────────────────────────────────────────────────┐              ║
║     │ Tools: Ag-Analytics API + Climate FieldView models   │              ║
║     │ Inputs: NDVI time-series, weather, historical yield  │              ║
║     │ Output: Field-level estimate 6 weeks pre-harvest     │              ║
║     │ Use: Grain contract timing, logistics planning       │              ║
║     └────────────────────────┬────────────────────────────┘              ║
║                              │                                           ║
║  6. HARVEST                  ▼                                           ║
║     ┌─────────────────────────────────────────────────────┐              ║
║     │ Guidance: Trimble or John Deere AutoTrac             │              ║
║     │ Yield monitoring: Ops Center yield map logging       │              ║
║     │ Logistics: Granular harvest tracking                 │              ║
║     └────────────────────────┬────────────────────────────┘              ║
║                              │                                           ║
║  7. POST-HARVEST ANALYSIS    ▼                                           ║
║     ┌─────────────────────────────────────────────────────┐              ║
║     │ Tools: Climate FieldView profit mapping              │              ║
║     │ Review: Yield vs. input cost per zone                │              ║
║     │ Feed into: Next season's soil sampling plan          │              ║
║     │ Carbon tracking: Regrow / Indigo Carbon              │              ║
║     └─────────────────────────────────────────────────────┘              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Platform Deep Dives

### John Deere Operations Center

The Operations Center (operations.deere.com) is the central data hub for John Deere's connected equipment ecosystem, serving over 350 million acres of data globally. It functions as the connective tissue between field machines, agronomic software, and farm management — effectively the ERP of a modern farm operation.

**Key features:**

- **Machine connectivity:** Automatic upload of planting, spraying, and harvest data from JD-compatible equipment via JDLink telematics
- **Field health maps:** Satellite and drone imagery overlaid with yield history and soil data for comparative field analysis
- **Prescription management:** Receives zone-based VRA prescription maps from Climate FieldView, Granular, or third-party agronomists and pushes them to machine controllers
- **Operations calendar and compliance records:** Tracks field-level activities for regulatory reporting, crop insurance documentation, and sustainability audits
- **API ecosystem:** Open MyJohnDeere API allows agronomic software vendors to read/write field data, enabling third-party platform interoperability
- **LLM integration (2024–):** Natural language interface allows farmers to query operational data without learning the UI

The Operations Center is at its most powerful when combined with a John Deere fleet — but its APIs make it accessible as a data destination for farms using mixed-brand equipment.

---

### Climate FieldView (Bayer)

Climate FieldView (climate.com) is the largest precision agriculture data platform in North America, with over 180 million connected acres. Acquired by Bayer (via Monsanto) in 2017, it has become the dominant agronomic software stack for row crop farmers in the US Corn Belt.

**Key features:**

- **Digital field maps:** Drag-and-drop field boundary creation; automatic sync with connected planters and combines
- **In-season imagery:** Weekly satellite NDVI imagery overlaid with historical yield maps; anomaly alerts sent via mobile
- **Nitrogen advisor:** ML model combining satellite greenness, soil type, weather, and hybrid data to generate variable-rate N top-dress recommendations — the highest-ROI feature for corn growers
- **Harvest dashboard:** Real-time combine yield data streamed to mobile; season-long yield map built as harvest progresses
- **FieldView Drive:** $99 hardware dongle retrofits non-connected older equipment for data capture
- **Profit mapping:** Overlays per-field input costs with yield revenue to identify unprofitable zones
- **Agronomist sharing:** Fields and data can be shared with a crop advisor or agronomist for remote consultation

FieldView's freemium model (basic imagery and field maps free; advanced features subscription) has driven extremely high adoption and made satellite-informed farming accessible to operations of all sizes.

---

### Taranis

Taranis (acquired by Indigo Agriculture, taranis.ag) is the highest-resolution AI scouting platform in agriculture — capturing sub-centimetre imagery from aircraft and drones to detect crop threats at a scale and precision impossible with any other method.

**Key features:**

- **Ultra-high-resolution aerial capture:** Proprietary aircraft with specialised sensors capture imagery at 0.1–1 mm ground sample distance — detailed enough to identify individual insect life stages on leaves
- **AI detection engine:** Deep learning models trained on millions of labelled crop images identify over 200 pest species, 150 disease signatures, and nutrient deficiencies with field-location precision
- **Issue mapping:** Detections are georeferenced and displayed on a web dashboard as heatmaps showing infestation density and distribution
- **Prescription integration:** Detected pest zones can be exported as variable-rate prescription maps for targeted spray application
- **Trend monitoring:** Multi-visit data tracked over the season to show whether interventions are working and whether threats are spreading
- **Agronomist alerts:** Threshold-based notifications trigger when pest or disease pressure exceeds user-defined action thresholds
- **Integration with Ops Center / FieldView:** Taranis detection maps can be layered alongside yield history and satellite NDVI for contextualised decision-making

Taranis is most commonly deployed by large-scale row crop and specialty crop operations where the economics of early, precise intervention justify the service cost.

---

## ROI & Metrics

| Use Case | Average Improvement | Source |
|---|---|---|
| Yield increase (precision N management) | +5–15% | McKinsey Global Institute, 2020 |
| Water savings (AI-driven irrigation) | 20–50% reduction | FAO AQUASTAT, 2022 |
| Herbicide reduction (See & Spray / targeted) | 50–77% reduction | Blue River Technology / John Deere |
| Pesticide use reduction (precision scouting) | 15–30% reduction | Taranis case studies |
| Labour savings (autonomous guidance + monitoring) | 20–30% reduction | USDA ERS, 2021 |
| Input cost savings (variable-rate seeding/fert) | $25–65/acre | Climate FieldView customer data |
| Early disease detection (Plantix vs. visual) | 2–5 days earlier | PEAT GmbH research |
| Yield forecast accuracy (6 weeks out) | ±8–12% vs. ±20–25% manual | Ag-Analytics platform documentation |
| Carbon credit generation (soil carbon practices) | $10–40/tonne CO₂e | Indigo Carbon / Regrow reports |

---

## Getting Started Guide

This is a step-by-step adoption path for a farm wanting to move from paper records to AI-assisted precision agriculture — no coding required.

### Step 1: Start with Free Satellite Monitoring (Week 1)

**Tool: OneSoil (onesoil.ai) — free**

1. Go to onesoil.ai and create a free account
2. Draw your field boundaries on the map (or upload a shapefile)
3. View weekly NDVI maps going back several seasons
4. Identify fields with consistent low-productivity zones — these are your first precision management targets

**What you learn:** Which fields have spatial variability worth managing. No hardware required.

---

### Step 2: Connect Your Equipment (Week 2–4)

**Tool: John Deere Operations Center (free basic tier) or Climate FieldView Drive ($99 dongle)**

- If you run John Deere equipment with JDLink, activate the Operations Center and verify field data is uploading automatically
- If you run older or mixed equipment, use a FieldView Drive dongle on each machine to capture planting and harvest data digitally

**Data requirement:** At minimum, one season of yield monitor data dramatically improves all downstream AI analysis. Start capturing it now if you have not already.

---

### Step 3: Add Soil Sensing on Your Most Variable Field (Month 2)

**Tool: CropX or Arable Mark sensor installation**

- Contact a local CropX dealer or Arable reseller
- Install 2–4 sensor nodes per field (they handle installation)
- Connect the sensor platform to your Operations Center or FieldView account via their integration
- Let the system collect a baseline of 2–4 weeks before acting on recommendations

**Expected outcome:** You will receive mobile irrigation alerts rather than irrigating on a calendar schedule — typically reducing water use 20–30% in the first season.

---

### Step 4: Add Disease Scouting to Your Workflow (Ongoing)

**Tool: Plantix (free smartphone app)**

- Download Plantix on Android or iOS
- When you see a suspect plant, photograph the symptomatic tissue (leaf, stem, or fruit)
- Review the AI diagnosis and recommended treatment
- Log your scouting observations in the app — they build a field-level pest pressure history over time

**Upgrade path:** If scouting becomes a bottleneck (large acreage, fast-moving threats), evaluate Taranis for aerial scouting as a managed service.

---

### Step 5: Generate Your First Prescription Map (Month 3–4)

**Tool: Climate FieldView or Farmers Edge**

- Once you have at least one season of yield monitor data in FieldView, navigate to the Nitrogen Advisor tool
- Review the variable-rate N recommendation map it generates for your top fields
- Export the prescription to your Operations Center or Trimble display
- Apply variable-rate and document the application in FieldView

**Data connections needed:** FieldView ↔ Operations Center integration (one-click OAuth in FieldView settings).

---

### Step 6: Evaluate ROI and Plan Next Season

**Tool: Climate FieldView Profit Mapping + Granular**

- After harvest, overlay your yield map against input application maps to calculate per-zone profitability
- Identify the bottom 20% of field zones by return on input — these are candidates for reduced input rates or alternative management
- Use Granular for whole-farm profit and loss tracking against benchmarks

---

## Compliance & Sustainability

### Data Ownership Concerns

Farm data is among the most sensitive business data a producer generates — it reveals productivity, input costs, agronomic decisions, and ultimately farm profitability. Key concerns:

- **Who owns the data?** Most platform terms of service grant the farm operator ownership of their data, but grant the platform a licence to use anonymised/aggregated data for model training. The **Farm Bureau's Privacy and Security Principles for Farm Data** (2014) provides a voluntary framework that platforms like FieldView and Granular have signed.
- **Data portability:** Check that your platform supports standard export formats (shapefiles, ISO-XML prescription files) so you can migrate to another provider without losing your agronomic history.
- **Third-party sharing:** Be explicit about whether your agronomic data is being shared with input suppliers, insurers, or commodity traders — some platforms offer tiered sharing settings.
- **EU GDPR:** European farms operating under GDPR should ensure their platform provider operates EU data residency or has adequate data processing agreements (DPAs) in place.

---

### EU AI Act Implications for Agriculture

The EU AI Act (in force from August 2024, with phased compliance timelines) classifies most agricultural AI applications as **limited or minimal risk** — they do not fall into the high-risk categories (which focus on safety-critical and fundamental rights applications). However, specific scenarios attract scrutiny:

- **Automated subsidy and insurance decisions** based on AI-generated crop assessments may constitute high-risk AI use under Annex III if they significantly affect farm income
- **Autonomous machinery operating in public spaces** (e.g., autonomous tractors on public roads) may require conformity assessments
- Farmers using AI tools procured from EU-regulated suppliers should request AI transparency documentation (technical documentation, accuracy claims, intended use statements) from vendors as part of procurement

Practically, most farms will not face direct EU AI Act compliance obligations — but large agri-businesses deploying AI at scale in EU markets should review their supplier agreements.

---

### Carbon Footprint Tracking & Sustainability Tools

Carbon markets and sustainability reporting are creating a new revenue stream and compliance requirement for farms. Three platforms lead this space:

| Platform | Approach | Key Feature | Revenue Model |
|---|---|---|---|
| **Regrow** (regrow.ag) | Soil carbon + methane modelling | Integration with FieldView & Ops Center; MRV for carbon protocols | SaaS + credit marketplace |
| **Agreena** (agreena.com) | Regenerative agriculture certification | EU-focused, soil sample verification, carbon credits | Credit revenue share |
| **Indigo Carbon** (indigoag.com) | Practice-based carbon payment | Pay-per-acre for cover crop, no-till, reduced fertiliser | Upfront payment per tonne |

These platforms connect directly to your existing farm management data (planting records, tillage logs, fertiliser applications) to minimise the additional data entry burden for farmers.

**Sustainability reporting:** For agri-businesses preparing Scope 3 emissions reports under CSRD (EU) or SEC climate disclosure rules (US), platforms like **Regrow** and **Farmers Business Network (FBN)** offer supply-chain-level emissions data aggregation from connected farms.

---

## References

1. McKinsey Global Institute (2020). *Precision farming: Improving crop yields and cutting resource use.* McKinsey & Company. https://www.mckinsey.com/industries/agriculture/our-insights/agriculture-and-climate-change

2. FAO (2022). *The State of Food and Agriculture 2022: Leveraging Automation in Agriculture.* Food and Agriculture Organization of the United Nations. https://www.fao.org/publications/sofa/2022/en/

3. USDA Economic Research Service (2021). *Precision Agriculture in the Digital Era: Recent Adoption on US Farms.* ERS Report No. EIB-231. https://www.ers.usda.gov/webdocs/publications/102025/eib-231.pdf

4. Kamilaris, A. & Prenafeta-Boldú, F.X. (2018). Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture*, 147, 70–90. https://doi.org/10.1016/j.compag.2018.02.016

5. Liakos, K.G., Busato, P., Moshou, D., Pearson, S. & Bochtis, D. (2018). Machine learning in agriculture: A review. *Sensors*, 18(8), 2674. https://doi.org/10.3390/s18082674

6. Wolfert, S., Ge, L., Verdouw, C. & Bogaardt, M.J. (2017). Big data in smart farming — A review. *Agricultural Systems*, 153, 69–80. https://doi.org/10.1016/j.agsy.2017.01.023

7. Blue River Technology / John Deere (2023). *See & Spray: Precision weed control for corn and cotton.* Official product documentation. https://www.bluerivertechnology.com/see-spray-ultimate/

8. PEAT GmbH (2023). *Plantix impact report: Crop disease detection at scale.* https://plantix.net/en/impact

9. John Deere (2024). *Operations Center and MyJohnDeere API developer documentation.* https://developer.deere.com/

10. MarketsandMarkets (2024). *AI in Agriculture Market — Global Forecast to 2028.* Report AG 8276. https://www.marketsandmarkets.com/Market-Reports/ai-in-agriculture-market-159957499.html

11. Lobell, D.B. & Asseng, S. (2017). Comparing estimates of climate change impacts from process-based and statistical crop models. *Nature Climate Change*, 7(1), 27–31. https://doi.org/10.1038/nclimate3200

12. European Commission (2024). *EU AI Act — Regulation (EU) 2024/1689 of the European Parliament and of the Council.* Official Journal of the EU. https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
