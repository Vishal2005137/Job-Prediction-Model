import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_store.pkl")

COMPANY_OPTIONS = [
    "Infosys",
    "TCS",
    "Wipro",
    "HCL Technologies",
    "IBM India",
    "Deloitte",
    "EY India",
    "Capgemini",
    "JP Morgan India",
    "Amazon India",
    "Flipkart",
    "Zoho",
    "Razorpay",
    "Uber",
    "Shiprocket",
    "Bread Financial",
    "Tesco",
    "Groww",
    "Siemens",
    "Sanofi",
    "Microsoft India",
    "Google India",
    "Juspay",
    "Accenture",
    "Reliance Industry",
    "SBI",
    "Maruti Suzuki",
    "Tata Motors",
    "Bajaj Finance",
    "LIC",
]
DOMAIN_OPTIONS = [
    "Total",
    "AI/ML",
    "CyberSecurity",
    "Cloud Computing",
    "Data Science & Analytics",
    "Software Dev",
    "DevOps & SRE",
    "Blockchain & Web3",
    "IoT",
    "AR/VR & Metaverse",
    "Quantum Computing",
]
WORK_MODE_OPTIONS = ["Remote", "Hybrid", "Onsite"]

def load_model_store(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model store not found at {model_path}. Please run train_model.py first."
        )
    store = joblib.load(model_path)
    return store

def predict_vacancies(store, company, chosen_domain, year, work_mode, salary_lpa):
    models = store["models"]
    if chosen_domain == "Total":
        key = "total"
    else:
        key = chosen_domain

    if key not in models:
        raise ValueError(f"Model for '{key}' not available. Retrain the models.")

    info = models[key]
    pipe = info["pipeline"]
    feature_columns = info["feature_columns"]

    row = {
        "Company Name": company,
        "Year": int(year),
        "Work Mode": work_mode,
        "Highest Salary (INR LPA)": float(salary_lpa),
    }

    if "Domain" in feature_columns:
        row["Domain"] = chosen_domain if chosen_domain != "Total" else store["models"]["total"]["feature_columns"][ -1]
        # If Total model expects Domain, set an informative value; using selected domain is fine

    X = pd.DataFrame([row], columns=feature_columns)
    pred = pipe.predict(X)[0]
    return max(0, round(float(pred)))

class VacancyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Job Vacancy Predictor")
        self.geometry("520x360")
        self.resizable(False, False)

        try:
            self.store = load_model_store()
        except Exception as e:
            messagebox.showerror("Model Error", str(e))
            self.destroy()
            return

        content = ttk.Frame(self, padding=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Company
        ttk.Label(content, text="Company Name").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.company_var = tk.StringVar(value=COMPANY_OPTIONS[0])
        ttk.Combobox(content, textvariable=self.company_var, values=COMPANY_OPTIONS, state="readonly").grid(row=0, column=1, sticky=tk.EW)

        # Domain/Target
        ttk.Label(content, text="Target (Total or Domain)").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.domain_var = tk.StringVar(value=DOMAIN_OPTIONS[0])
        ttk.Combobox(content, textvariable=self.domain_var, values=DOMAIN_OPTIONS, state="readonly").grid(row=1, column=1, sticky=tk.EW)

        # Year
        ttk.Label(content, text="Year (2025-2030)").grid(row=2, column=0, sticky=tk.W, pady=6)
        self.year_var = tk.StringVar(value="2025")
        ttk.Spinbox(content, from_=2025, to=2030, textvariable=self.year_var, width=10).grid(row=2, column=1, sticky=tk.W)

        # Work Mode
        ttk.Label(content, text="Work Mode").grid(row=3, column=0, sticky=tk.W, pady=6)
        self.workmode_var = tk.StringVar(value=WORK_MODE_OPTIONS[1])
        ttk.Combobox(content, textvariable=self.workmode_var, values=WORK_MODE_OPTIONS, state="readonly").grid(row=3, column=1, sticky=tk.EW)

        # Salary
        ttk.Label(content, text="Highest Salary (INR LPA)").grid(row=4, column=0, sticky=tk.W, pady=6)
        self.salary_var = tk.StringVar(value="15")
        ttk.Entry(content, textvariable=self.salary_var).grid(row=4, column=1, sticky=tk.EW)

        # Predict button
        predict_btn = ttk.Button(content, text="Predict", command=self.on_predict)
        predict_btn.grid(row=5, column=0, columnspan=2, pady=12, sticky=tk.EW)

        # Result
        self.result_var = tk.StringVar(value="")
        self.result_msg = tk.Message(
            content,
            textvariable=self.result_var,
            width=480,
            justify="center",
            anchor="center",
            font=("Segoe UI", 12, "bold"),
        )
        self.result_msg.grid(row=6, column=0, columnspan=2, pady=8, sticky=tk.EW)

        for i in range(2):
            content.columnconfigure(i, weight=1)

    def on_predict(self):
        try:
            pred = predict_vacancies(
                self.store,
                self.company_var.get(),
                self.domain_var.get(),
                int(self.year_var.get()),
                self.workmode_var.get(),
                float(self.salary_var.get()),
            )
            target_label = self.domain_var.get()
            company = self.company_var.get()
            year = self.year_var.get()
            if target_label == "Total":
                msg = f"Predicted Total Vacancies for {company} {year} = {pred}"
            else:
                msg = f"Predicted Vacancies in {target_label} for {company} {year} = {pred}"
            self.result_var.set(msg)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

import threading

def run_app():
    app = VacancyApp()
    app.mainloop()

if __name__ == "__main__":
    # This allows running the app as a standalone script
    run_app()