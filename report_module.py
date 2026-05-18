import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

def generate_pdf_report(transactions):
    df = pd.DataFrame(transactions)
    summary = df.groupby("category")["amount"].sum()

    plt.figure(figsize=(6, 6))
    summary.plot.pie(autopct='%1.1f%%')
    plt.title("Spending by Category")
    chart_path = "category_pie.png"
    plt.savefig(chart_path)
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Budget Report", ln=True, align='C')
    pdf.image(chart_path, x=30, y=30, w=150)
    output_path = "budget_report.pdf"
    pdf.output(output_path)

    return output_path
