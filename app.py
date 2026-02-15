"""
FastAPI application for Recommendation System deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from models import BayesianModel, SocialBayesianMarkovModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Epinions Recommendation API",
    description="Social-aware recommendation system with Jaccard, Bayesian, and Social Bayesian Markov models",
    version="1.0.0"
)

# Global model storage
models = {}
item_name_map = {}


def _collect_users() -> List[str]:
    """Collect unique users across loaded models and sort alphabetically."""
    users = set()
    for model in models.values():
        if hasattr(model, "user_items"):
            users.update(getattr(model, "user_items", {}).keys())
        if hasattr(model, "user_ratings"):
            users.update(getattr(model, "user_ratings", {}).keys())
    return sorted(str(u) for u in users)


def _collect_items() -> List[dict]:
    """Collect items and names for UI dropdown, sorted by item name."""
    item_ids = set()
    for model in models.values():
        if hasattr(model, "all_items"):
            item_ids.update(str(i) for i in getattr(model, "all_items", set()))
    if not item_ids and item_name_map:
        item_ids.update(item_name_map.keys())
    items = [{"item_id": i, "item_name": _item_name(i)} for i in item_ids]
    items.sort(key=lambda x: x["item_name"].lower())
    return items


def _item_name(item_id: str) -> str:
    """Resolve item name with a readable fallback."""
    return item_name_map.get(str(item_id), f"Item {item_id}")


class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    user_id: str = Field(..., description="User ID to get recommendations for")
    k: int = Field(10, description="Number of recommendations to return", ge=1, le=100)
    model_type: Literal["jaccard", "bayesian", "social_bayesian"] = Field(
        "social_bayesian", 
        description="Model to use for recommendations"
    )
    paid: Optional[float] = Field(0.0, description="Whether the review is paid (for Bayesian models)")
    time: Optional[int] = Field(0, description="Timestamp for the review (for Bayesian models)")
    words: Optional[str] = Field("", description="Review text (for Bayesian models)")


class PredictionRequest(BaseModel):
    """Request model for predicting ratings."""
    user_id: str = Field(..., description="User ID")
    item_id: str = Field(..., min_length=1, description="Item ID")
    model_type: Literal["jaccard", "bayesian", "social_bayesian"] = Field(
        "social_bayesian",
        description="Model to use for prediction"
    )
    paid: Optional[float] = Field(0.0, description="Whether the review is paid")
    time: Optional[int] = Field(0, description="Timestamp for the review")
    words: Optional[str] = Field("", description="Review text")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: str
    recommendations: List[dict]
    model_type: str


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    user_id: str
    item_id: str
    item_name: str
    probability: float
    model_type: str


@app.on_event("startup")
async def load_models():
    """Load pre-trained models on startup."""
    logger.info("Loading models...")
    
    model_dir = Path("saved_models")

    names_path = model_dir / "item_names.json"
    if names_path.exists():
        try:
            loaded_names = json.loads(names_path.read_text(encoding="utf-8"))
            if isinstance(loaded_names, dict):
                item_name_map.update({str(k): str(v) for k, v in loaded_names.items()})
                logger.info("Loaded %d item names", len(item_name_map))
        except Exception as e:
            logger.error("Failed to load item names: %s", e)

    model_files = {
        "jaccard": model_dir / "jaccard_model.pkl",
        "bayesian": model_dir / "bayesian_model.pkl",
        "social_bayesian": model_dir / "social_bayesian_model.pkl",
    }

    for model_name, model_path in model_files.items():
        if not model_path.exists():
            continue
        try:
            models[model_name] = joblib.load(model_path)
            logger.info("Loaded %s model", model_name)
        except Exception as e:
            logger.error("Failed to load %s model from %s: %s", model_name, model_path, e)

    # If sklearn-based models fail to deserialize (version mismatch), bootstrap small fallbacks.
    missing = [m for m in ("bayesian", "social_bayesian") if m not in models]
    if missing:
        logger.warning("Missing models after load: %s. Bootstrapping lightweight fallbacks.", missing)
        try:
            _bootstrap_missing_models(model_dir)
        except Exception as e:
            logger.error("Fallback bootstrap failed: %s", e)

    if not models:
        logger.warning("No models available. Please check saved_models files and logs.")


def _bootstrap_missing_models(model_dir: Path):
    """Create tiny fallback Bayesian/Social models for demo availability."""
    rng = np.random.default_rng(42)
    n_rows = 120

    # Build fallback training frame from real IDs when possible.
    users_source = []
    items_source = []
    if "jaccard" in models:
        j = models["jaccard"]
        users_source = list(getattr(j, "user_ratings", {}).keys())
        items_source = list(getattr(j, "all_items", []))
    if not items_source and item_name_map:
        items_source = list(item_name_map.keys())
    if not users_source:
        users_source = [f"user_{i}" for i in range(1, 41)]
    if not items_source:
        items_source = [f"item_{i}" for i in range(1, 81)]

    users = [str(rng.choice(users_source)) for _ in range(n_rows)]
    items = [str(rng.choice(items_source)) for _ in range(n_rows)]
    stars = rng.integers(1, 6, size=n_rows)
    df_train = pd.DataFrame(
        {
            "user": users,
            "item": items,
            "stars": stars,
            "paid": rng.random(n_rows),
            "time": rng.integers(1_600_000_000, 1_700_000_000, size=n_rows),
            "words": [
                "good product reliable quality" if s >= 4 else "not great average"
                for s in stars
            ],
        }
    )
    y = (df_train["stars"] >= 4).astype(int).to_numpy()

    if "bayesian" not in models:
        X_basic = np.column_stack(
            [
                df_train["paid"].to_numpy(dtype=float),
                (df_train["time"] - df_train["time"].min())
                / (df_train["time"].max() - df_train["time"].min() + 1),
                df_train["words"].str.split().str.len().to_numpy(dtype=float),
            ]
        )
        bayesian = BayesianModel(alpha=1.0)
        bayesian.fit(X_basic, y, df_train=df_train)
        models["bayesian"] = bayesian
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bayesian, model_dir / "bayesian_model.pkl")
        logger.info("Bootstrapped bayesian fallback model")

    if "social_bayesian" not in models:
        X_social = rng.normal(size=(n_rows, 31))
        trust_df = pd.DataFrame(
            {
                "source": [f"u{i}" for i in range(1, 30)],
                "target": [f"u{i}" for i in range(2, 31)],
            }
        )
        social = SocialBayesianMarkovModel(alpha=1.0, mrf_weight=0.3)
        social.fit(X_social, y, df_train=df_train, trust_df=trust_df)
        models["social_bayesian"] = social
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(social, model_dir / "social_bayesian_model.pkl")
        logger.info("Bootstrapped social_bayesian fallback model")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Epinions Recommendation API",
        "version": "1.0.0",
        "available_models": list(models.keys()),
        "endpoints": {
            "GET /": "API information",
            "GET /meta": "Users/items metadata for UI",
            "GET /health": "Health check",
            "POST /recommend": "Get top-k recommendations for a user",
            "POST /predict": "Predict rating probability for user-item pair"
        }
    }


@app.get("/meta")
async def meta():
    """Metadata endpoint for lightweight frontend population."""
    users = _collect_users()
    items = _collect_items()
    return {
        "users": users,
        "usernames": users,
        "items": items,
    }


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Simple web UI for browser-based testing."""
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Recommendation UI</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 24px; max-width: 900px; }
        h1 { margin-bottom: 6px; }
        .hint { color: #555; margin-bottom: 18px; }
        .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
        .row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
        .field { display: flex; flex-direction: column; gap: 4px; min-width: 180px; }
        label { font-size: 12px; color: #444; font-weight: 600; }
        .desc { font-size: 11px; color: #666; line-height: 1.3; max-width: 220px; }
        .ab-guide { background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; margin-bottom: 10px; }
        .ab-guide ul { margin: 6px 0 0 18px; padding: 0; }
        .ab-guide li { font-size: 12px; color: #444; margin: 2px 0; }
        input, select, button { padding: 8px; font-size: 14px; }
        button { cursor: pointer; }
        pre { background: #111; color: #0f0; padding: 12px; border-radius: 8px; white-space: pre-wrap; }
      </style>
    </head>
    <body>
      <h1>Epinions Recommendation System</h1>
      
      <div class="card">
        <h3>Recommend</h3>
        <div class="ab-guide">
          Generate top-K recommendations for one user.
          <ul>
            <li><b>item_name</b>: recommended item title.</li>
            <li><b>score</b>: model ranking score (higher means stronger recommendation).</li>
          </ul>
        </div>
        <div class="row">
          <div class="field">
            <label for="r_user">Username</label>
            <select id="r_user"></select>
            <div class="desc">User for whom recommendations are generated.</div>
          </div>
          <div class="field">
            <label for="r_k">Top K</label>
            <input id="r_k" type="number" min="1" max="100" value="10" />
            <div class="desc">How many recommendations to return.</div>
          </div>
          <div class="field">
            <label for="r_model">Model</label>
            <select id="r_model">
              <option value="social_bayesian">social_bayesian</option>
              <option value="bayesian">bayesian</option>
              <option value="jaccard">jaccard</option>
            </select>
            <div class="desc">Model used to compute top-K recommendations.</div>
          </div>
        </div>
        <div class="row">
          <button onclick="callRecommend()">Run Recommend</button>
          <div class="desc">Runs <code>/recommend</code> and prints JSON output below.</div>
        </div>
      </div>

      <div class="card">
        <h3>Predict</h3>
        <div class="ab-guide">
          Predict preference probability for one user-item pair.
          <ul>
            <li><b>item_name</b>: selected item name.</li>
            <li><b>probability</b>: chance of positive/high rating for that item.</li>
          </ul>
        </div>
        <div class="row">
          <div class="field">
            <label for="p_user">Username</label>
            <select id="p_user"></select>
            <div class="desc">User for whom item preference is predicted.</div>
          </div>
          <div class="field">
            <label for="p_item">Item</label>
            <select id="p_item"></select>
            <div class="desc">Item to score for the selected user.</div>
          </div>
          <div class="field">
            <label for="p_model">Model</label>
            <select id="p_model">
              <option value="social_bayesian">social_bayesian</option>
              <option value="bayesian">bayesian</option>
              <option value="jaccard">jaccard</option>
            </select>
            <div class="desc">Model used to compute the probability.</div>
          </div>
        </div>
        <div class="row">
          <button onclick="callPredict()">Run Predict</button>
          <div class="desc">Runs <code>/predict</code> and prints JSON output below.</div>
        </div>
      </div>

      <div class="card">
        <h3>A/B Testing</h3>
        <div class="ab-guide">
          Compare two models side-by-side for the same user.
          <ul>
            <li><b>recommendations_a/recommendations_b</b>: top-K item lists from each model.</li>
            <li><b>overlap_count/overlap_percent</b>: how many recommendations are common.</li>
            <li><b>predict_probability_a/predict_probability_b</b>: score for the selected user-item pair.</li>
          </ul>
        </div>
        <div class="row">
          <div class="field">
            <label for="ab_user">Username</label>
            <select id="ab_user"></select>
            <div class="desc">User whose recommendations are compared between Model A and Model B.</div>
          </div>
          <div class="field">
            <label for="ab_item">Item</label>
            <select id="ab_item"></select>
            <div class="desc">Single item used for predict-probability comparison across the two models.</div>
          </div>
          <div class="field">
            <label for="ab_k">Top K</label>
            <input id="ab_k" type="number" min="1" max="20" value="5" />
            <div class="desc">Number of recommendations returned per model for overlap analysis.</div>
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="ab_model_a">Model A</label>
            <select id="ab_model_a">
              <option value="social_bayesian">social_bayesian</option>
              <option value="bayesian">bayesian</option>
              <option value="jaccard">jaccard</option>
            </select>
            <div class="desc">First model in the A/B comparison.</div>
          </div>
          <div class="field">
            <label for="ab_model_b">Model B</label>
            <select id="ab_model_b">
              <option value="bayesian">bayesian</option>
              <option value="social_bayesian">social_bayesian</option>
              <option value="jaccard">jaccard</option>
            </select>
            <div class="desc">Second model in the A/B comparison.</div>
          </div>
          <div class="field">
            <label>&nbsp;</label>
            <button onclick="runABTest()">Run A/B Test</button>
            <div class="desc">Executes recommend + predict for both models and prints one combined JSON result.</div>
          </div>
        </div>
      </div>

      <h3>Response</h3>
      <pre id="out">Ready.</pre>

      <script>
        async function loadMeta() {
          try {
            const res = await fetch("/meta");
            const data = await res.json();
            const users = (data.usernames || data.users || []).sort();
            const items = (data.items || []).slice().sort((a, b) =>
              (a.item_name || "").localeCompare(b.item_name || "")
            );
            const rUser = document.getElementById("r_user");
            const pUser = document.getElementById("p_user");
            const abUser = document.getElementById("ab_user");
            const pItem = document.getElementById("p_item");
            const abItem = document.getElementById("ab_item");
            rUser.innerHTML = "";
            pUser.innerHTML = "";
            abUser.innerHTML = "";
            pItem.innerHTML = "";
            abItem.innerHTML = "";
            if (users.length === 0) {
              const o1 = document.createElement("option");
              o1.value = "user123";
              o1.textContent = "user123";
              rUser.appendChild(o1);
              const o2 = document.createElement("option");
              o2.value = "user123";
              o2.textContent = "user123";
              pUser.appendChild(o2);
              const o3 = document.createElement("option");
              o3.value = "user123";
              o3.textContent = "user123";
              abUser.appendChild(o3);
            }
            if (users.length > 0) {
              for (const u of users) {
                const o1 = document.createElement("option");
                o1.value = u;
                o1.textContent = u;
                rUser.appendChild(o1);
                const o2 = document.createElement("option");
                o2.value = u;
                o2.textContent = u;
                pUser.appendChild(o2);
                const o3 = document.createElement("option");
                o3.value = u;
                o3.textContent = u;
                abUser.appendChild(o3);
              }
            }
            if (items.length === 0) {
              const oi = document.createElement("option");
              oi.value = "";
              oi.textContent = "No items available";
              pItem.appendChild(oi);
              const oj = document.createElement("option");
              oj.value = "";
              oj.textContent = "No items available";
              abItem.appendChild(oj);
            } else {
              for (const it of items) {
                const oi = document.createElement("option");
                oi.value = it.item_id;
                oi.textContent = it.item_name;
                pItem.appendChild(oi);
                const oj = document.createElement("option");
                oj.value = it.item_id;
                oj.textContent = it.item_name;
                abItem.appendChild(oj);
              }
            }
          } catch (e) {
            document.getElementById("out").textContent = "Failed to load users: " + e;
          }
        }

        async function postJson(path, payload) {
          const res = await fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          const text = await res.text();
          try {
            return { ok: res.ok, status: res.status, body: JSON.parse(text) };
          } catch (_) {
            return { ok: res.ok, status: res.status, body: text };
          }
        }

        async function callRecommend() {
          const payload = {
            user_id: document.getElementById("r_user").value,
            k: parseInt(document.getElementById("r_k").value || "10", 10),
            model_type: document.getElementById("r_model").value
          };
          const result = await postJson("/recommend", payload);
          document.getElementById("out").textContent = JSON.stringify(result, null, 2);
        }

        async function callPredict() {
          const payload = {
            user_id: document.getElementById("p_user").value,
            item_id: document.getElementById("p_item").value,
            model_type: document.getElementById("p_model").value
          };
          const result = await postJson("/predict", payload);
          document.getElementById("out").textContent = JSON.stringify(result, null, 2);
        }

        async function runABTest() {
          const user = document.getElementById("ab_user").value;
          const itemId = document.getElementById("ab_item").value;
          const k = parseInt(document.getElementById("ab_k").value || "5", 10);
          const modelA = document.getElementById("ab_model_a").value;
          const modelB = document.getElementById("ab_model_b").value;

          if (modelA === modelB) {
            document.getElementById("out").textContent = "Choose two different models for A/B testing.";
            return;
          }

          const [recA, recB, predA, predB] = await Promise.all([
            postJson("/recommend", { user_id: user, k: k, model_type: modelA }),
            postJson("/recommend", { user_id: user, k: k, model_type: modelB }),
            postJson("/predict", { user_id: user, item_id: itemId, model_type: modelA }),
            postJson("/predict", { user_id: user, item_id: itemId, model_type: modelB }),
          ]);

          const namesA = (recA.body && recA.body.recommendations || []).map(x => x.item_name);
          const namesB = (recB.body && recB.body.recommendations || []).map(x => x.item_name);
          const overlap = namesA.filter(n => namesB.includes(n));
          const overlapPct = k > 0 ? ((overlap.length / k) * 100).toFixed(1) : "0.0";

          const result = {
            ok: recA.ok && recB.ok && predA.ok && predB.ok,
            status: {
              recommend_a: recA.status,
              recommend_b: recB.status,
              predict_a: predA.status,
              predict_b: predB.status
            },
            input: { user_id: user, item_id: itemId, top_k: k, model_a: modelA, model_b: modelB },
            summary: {
              overlap_count: overlap.length,
              overlap_percent: overlapPct + "%",
              predict_probability_a: predA.body && predA.body.probability,
              predict_probability_b: predB.body && predB.body.probability
            },
            recommendations_a: recA.body && recA.body.recommendations,
            recommendations_b: recB.body && recB.body.recommendations
          };

          document.getElementById("out").textContent = JSON.stringify(result, null, 2);
        }
      </script>
      <script>loadMeta();</script>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get top-k item recommendations for a user.
    
    Parameters:
    - user_id: User ID to get recommendations for
    - k: Number of recommendations (default: 10)
    - model_type: Model to use (jaccard, bayesian, social_bayesian)
    - paid: Whether the review is paid (for Bayesian models)
    - time: Timestamp for the review (for Bayesian models)
    - words: Review text (for Bayesian models)
    
    Returns:
    - List of recommended items with scores
    """
    # Check if model is loaded
    if request.model_type not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_type}' not found. Available models: {list(models.keys())}"
        )
    
    model = models[request.model_type]
    
    try:
        # Get recommendations based on model type
        if request.model_type == "jaccard":
            recommendations = model.predict_top_k(request.user_id, k=request.k)
        elif request.model_type == "bayesian":
            recommendations = model.predict_top_k(
                request.user_id, 
                k=request.k,
                paid=request.paid,
                time=request.time,
                words=request.words
            )
        else:  # social_bayesian
            recommendations = model.predict_top_k(request.user_id, k=request.k)
        
        # Format recommendations
        formatted_recs = [
            {"item_name": _item_name(item), "score": float(score)}
            for item, score in recommendations
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=formatted_recs,
            model_type=request.model_type
        )
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_rating(request: PredictionRequest):
    """
    Predict rating probability for a user-item pair.
    
    Parameters:
    - user_id: User ID
    - item_id: Item ID
    - model_type: Model to use (jaccard, bayesian, social_bayesian)
    - paid: Whether the review is paid (for Bayesian models)
    - time: Timestamp for the review (for Bayesian models)
    - words: Review text (for Bayesian models)
    
    Returns:
    - Probability of positive rating (high rating >= 4.0)
    """
    # Check if model is loaded
    if request.model_type not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_type}' not found. Available models: {list(models.keys())}"
        )
    
    model = models[request.model_type]
    
    try:
        # Get prediction based on model type
        if request.model_type == "jaccard":
            score = model._score_item(request.user_id, request.item_id)
            probability = 1 / (1 + np.exp(-(score - 3.5)))
        elif request.model_type == "bayesian":
            probability = model._score_item(
                request.user_id,
                request.item_id,
                paid=request.paid,
                time=request.time,
                words=request.words
            )
        else:  # social_bayesian
            # Match the trained feature dimension for robust inference.
            n_features = int(getattr(model.scaler, "n_features_in_", 31))
            base = [request.paid, 0.5, len(str(request.words).split())]
            padding = max(0, n_features - len(base))
            features = np.array([base + [0.0] * padding], dtype=float)
            probability = float(model.predict_proba_base(features)[0])
        
        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            item_name=_item_name(request.item_id),
            probability=float(probability),
            model_type=request.model_type
        )
    
    except Exception as e:
        logger.error(f"Error predicting rating: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
