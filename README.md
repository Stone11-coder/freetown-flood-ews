\# Freetown Flood Early Warning System (EWS) 🌊📡



A production-grade, containerized Early Warning System (EWS) for real-time flood risk forecasting and emergency SMS alert dispatching across vulnerable catchments in Freetown, Sierra Leone.



\---



\## 📌 Executive Overview

Freetown's complex coastal-mountain topography and intense seasonal monsoons create severe flash flood vulnerabilities. This project transitions traditional static hydrogeological modeling into an automated, operational web service. 



The system processes dynamic rainfall forcing alongside spatial catchment parameters to predict flash flood probabilities. When predicted risk exceeds high-precision empirical decision thresholds, emergency bulletins are programmatically dispatched to alert personnel and municipal response units via automated SMS gateways.



\---



\## ⚙️ System Architecture



┌─────────────────────────┐

&#x20;                          │  Dynamic Rainfall Input │

&#x20;                          └────────────┬────────────┘

&#x20;                                       │

&#x20;                                       ▼

┌───────────────────────┐     ┌─────────────────────────┐

│ Catchment Morphometry │ ──> │ FastAPI Inference Engine│

└───────────────────────┘     │  (XGBoost v2.0.0 Model) │

└────────────┬────────────┘

│

▼

┌─────────────────────────┐

│ Youden's J Threshold    │

│  Optimization (0.9537)  │

└────────────┬────────────┘

│

\[ If Risk > 0.9537 ]

│

▼

┌─────────────────────────┐

│  Africa's Talking API   │

│  Automated SMS Dispatch │

└─────────────────────────┘





\---



\## 🛠️ Key Technical Features



\- \*\*Machine Learning Core\*\*: Built on \*\*XGBoost v2.0.0\*\*, calibrated using dynamic rainfall metrics, slope gradients, and hydrological flow paths.

\- \*\*Precision Thresholding\*\*: Optimized using \*\*Youden’s J Statistic ($J = \\text{Sensitivity} + \\text{Specificity} - 1$)\*\* with a cutoff threshold of \*\*$0.9537$\*\*, ensuring high sensitivity to genuine flood threats while eliminating false alarms.

\- \*\*Containerized FastAPI Service\*\*: Fully containerized using \*\*Docker\*\* and Linux-slim environments for seamless deployment on cloud providers (e.g., Render).

\- \*\*Automated SMS Dispatcher\*\*: Integrates HTTP client logic to format dynamic catchment bulletins and transmit live alerts to community leaders and disaster management teams.



\---



\## 💻 Tech Stack \& Dependencies



| Component | Tool / Framework |

| :--- | :--- |

| \*\*Language\*\* | Python 3.10 |

| \*\*API Backend\*\* | FastAPI, Uvicorn, Pydantic |

| \*\*ML Engine\*\* | XGBoost 2.0.0, Scikit-Learn, NumPy, Pandas |

| \*\*Alert Gateway\*\* | Africa's Talking SMS API via HTTP Requests |

| \*\*Containerization\*\* | Docker Desktop, Docker Engine |

| \*\*Cloud Deployment\*\* | Render Web Services |



\---



\## 🚀 Running Locally with Docker



\### 1. Build the Docker Container

```bash

docker build -t freetown-flood-ews:latest .



\---



\## 🛡️ License \& Intellectual Property Notice

\*\*Copyright © 2026 Stonison Tamba Kortu. All Rights Reserved.\*\*



\*This repository represents the public operational showcase of the Freetown Flood EWS framework. Raw DEM rasters, proprietary spatial feature engineering pipelines, and live production secrets are intentionally restricted.\*

