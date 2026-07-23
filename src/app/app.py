import os
import sys
from fastapi import FastAPI
import gradio as gr
from pydantic import BaseModel

# Ensure we can import from src/serving when running "uvicorn src.app.app:app"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict  # single source of truth for inference

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


# Request schema (unmodified)
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def api_predict(data: CustomerData):
    try:
        out = predict(data.model_dump() if hasattr(data, "model_dump") else data.dict())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}


# --- Gradio UI wrapper (unmodified payload mapping) ---
def gradio_interface(
    gender,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges,
):
    payload = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure) if tenure else 0,
        "MonthlyCharges": float(MonthlyCharges) if MonthlyCharges else 0.0,
        "TotalCharges": float(TotalCharges) if TotalCharges else 0.0,
    }
    out = predict(payload)

    # Format the prediction output cleanly for display
    if isinstance(out, dict):
        churn_risk = out.get("prediction", out)
        probability = out.get("probability", None)
        prob_str = f" ({probability*100:.1f}% risk)" if probability else ""
        return f"### Prediction Result: **{churn_risk}**{prob_str}"

    return f"### Prediction Result: **{out}**"


# --- Professional UI Layout using gr.Blocks ---
with gr.Blocks(title="Telco Churn Analytics") as demo:

    gr.Markdown(
        """
        # 📊 Telecom Customer Churn Prediction Engine
        *Enterprise Machine Learning Risk Assessment Dashboard*
        """
    )

    with gr.Row():
        # LEFT COLUMN: INPUT FORM (Organized in logical accordions)
        with gr.Column(scale=2):

            with gr.Accordion("👤 Demographics & Basic Info", open=True):
                with gr.Row():
                    gender = gr.Dropdown(
                        ["Male", "Female"], label="Gender", value="Male"
                    )
                    Partner = gr.Dropdown(
                        ["Yes", "No"], label="Has Partner?", value="No"
                    )
                    Dependents = gr.Dropdown(
                        ["Yes", "No"], label="Has Dependents?", value="No"
                    )

            with gr.Accordion("📞 Telecom & Line Services", open=True):
                with gr.Row():
                    PhoneService = gr.Dropdown(
                        ["Yes", "No"], label="Phone Service", value="Yes"
                    )
                    MultipleLines = gr.Dropdown(
                        ["Yes", "No", "No phone service"],
                        label="Multiple Lines",
                        value="No",
                    )
                    InternetService = gr.Dropdown(
                        ["DSL", "Fiber optic", "No"],
                        label="Internet Service",
                        value="Fiber optic",
                    )

            with gr.Accordion("🛡️ Add-On & Security Services", open=False):
                with gr.Row():
                    OnlineSecurity = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Online Security",
                        value="No",
                    )
                    OnlineBackup = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Online Backup",
                        value="No",
                    )
                    DeviceProtection = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Device Protection",
                        value="No",
                    )
                with gr.Row():
                    TechSupport = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Tech Support",
                        value="No",
                    )
                    StreamingTV = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Streaming TV",
                        value="No",
                    )
                    StreamingMovies = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Streaming Movies",
                        value="No",
                    )

            with gr.Accordion(
                "💳 Contract, Billing & Account Metrics", open=True
            ):
                with gr.Row():
                    Contract = gr.Dropdown(
                        ["Month-to-month", "One year", "Two year"],
                        label="Contract Type",
                        value="Month-to-month",
                    )
                    PaperlessBilling = gr.Dropdown(
                        ["Yes", "No"], label="Paperless Billing", value="Yes"
                    )
                    PaymentMethod = gr.Dropdown(
                        [
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)",
                        ],
                        label="Payment Method",
                        value="Electronic check",
                    )
                with gr.Row():
                    tenure = gr.Number(
                        label="Tenure (Months)", value=12, precision=0
                    )
                    MonthlyCharges = gr.Number(
                        label="Monthly Charges ($)", value=70.0
                    )
                    TotalCharges = gr.Number(
                        label="Total Charges ($)", value=840.0
                    )

            submit_btn = gr.Button(
                "🚀 Calculate Churn Risk", variant="primary", size="lg"
            )

        # RIGHT COLUMN: OUTPUT / PREDICTION PANEL
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Risk Assessment Output")
            output_box = gr.Markdown(
                value="*Submit customer details to compute assessment results.*"
            )

            gr.Info(
                "💡 **Tip:** Customers with month-to-month contracts and high monthly charges typically carry higher churn risks."
            )

    # Wire up button action
    submit_btn.click(
        fn=gradio_interface,
        inputs=[
            gender,
            Partner,
            Dependents,
            PhoneService,
            MultipleLines,
            InternetService,
            OnlineSecurity,
            OnlineBackup,
            DeviceProtection,
            TechSupport,
            StreamingTV,
            StreamingMovies,
            Contract,
            PaperlessBilling,
            PaymentMethod,
            tenure,
            MonthlyCharges,
            TotalCharges,
        ],
        outputs=output_box,
    )

app = gr.mount_gradio_app(app, demo, path="/ui")