# explain/dashboard_app.py
import streamlit as st
import joblib, os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.config import cfg
from core.labels import LABEL_MAP, REVERSE_LABEL_MAP

st.set_page_config(layout="wide", page_title="ML Guardian Demo")

@st.cache_resource
def load_models():
    try:
        xgb = joblib.load(cfg["paths"]["xgb_model"])
    except Exception as e:
        st.error(f"Failed to load XGBoost model: {e}")
        return None, None
    
    fusion = None
    fusion_path = cfg["paths"]["fusion_model"]
    if os.path.exists(fusion_path):
        try:
            # Check if file is too small (likely corrupted)
            file_size = os.path.getsize(fusion_path)
            if file_size < 1000:  # Less than 1KB is suspicious
                st.warning(f"Fusion model file is too small ({file_size} bytes), skipping...")
                # Remove the corrupted file
                try:
                    os.remove(fusion_path)
                    st.info("Corrupted fusion model file removed.")
                except:
                    pass
            else:
                fusion = joblib.load(fusion_path)
        except Exception as e:
            st.warning(f"Failed to load fusion model: {e}")
    else:
        st.info("No fusion model found. Using XGBoost model only.")
    return xgb, fusion

xgb_pipe, fusion = load_models()

# Allow hot-reloading models if new artifacts are saved while the app is running
if st.button("Reload models"):
    try:
        load_models.clear()
    except Exception:
        pass
    xgb_pipe, fusion = load_models()
    st.success("Models reloaded.")

if xgb_pipe is None:
    st.error("❌ No trained model available. Please run the training pipeline first.")
    st.stop()

st.title("ML Guardian — Hallucination Detector (Demo)")
claim = st.text_area("Claim", value="The Eiffel Tower is located in Paris.")
evidence = st.text_area("Evidence", value="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.")

if st.button("Predict"):
    # naive feature extraction using feature_engineer_optimized
    from core.feature_engineer_optimized import OptimizedFeatureEngineer
    fe = OptimizedFeatureEngineer()
    feats = fe.extract_all_features(claim, evidence)
    import pandas as pd
    X = pd.DataFrame([feats])
    # align feature order to pipeline
    fnames = xgb_pipe["feature_names"]
    X = X.reindex(columns=fnames, fill_value=0.0)
    probs = xgb_pipe["model"].predict_proba(X.values)[0]
    top_idx = int(np.argmax(probs))
    label = LABEL_MAP.get(top_idx, str(top_idx))
    st.metric("XGBoost Prediction", label)
    st.write("Probabilities:", {LABEL_MAP.get(i, str(i)): float(p) for i,p in enumerate(probs)})

    if fusion and "scaler" in fusion and "stacker" in fusion:
        try:
            # Build meta-features: start with XGB probs
            meta = probs.reshape(1, -1)
            # If fusion was trained with more features (e.g., XGB+MLP), pad zeros
            try:
                needed = int(getattr(fusion["scaler"], "n_features_in_", fusion["scaler"].mean_.shape[0]))
            except Exception:
                needed = meta.shape[1]
            if meta.shape[1] < needed:
                import numpy as np
                pad = np.zeros((1, needed - meta.shape[1]))
                meta = np.concatenate([meta, pad], axis=1)
            elif meta.shape[1] > needed:
                meta = meta[:, :needed]
            meta_s = fusion["scaler"].transform(meta)
            pred = fusion["stacker"].predict(meta_s)[0]
            st.metric("Ensemble Prediction", LABEL_MAP.get(int(pred), str(int(pred))))
        except Exception as e:
            st.warning(f"Ensemble prediction failed: {e}")
    else:
        st.info("Only XGBoost prediction available (no ensemble model)")

st.markdown("---")
st.markdown("**Notes:** This demo uses the optimized feature extractor to compute features for a single claim/evidence pair. For better accuracy, use the full pipeline (XGBoost + MLP + Transformer + Stacker).")