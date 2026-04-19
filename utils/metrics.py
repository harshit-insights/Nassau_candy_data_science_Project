import pandas as pd
import numpy as np


def calculate_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates and calculates all product-level profitability metrics.
    Used in: product ranking, Pareto, cost diagnostics.
    """
    product_margin = df.groupby('Product Name').agg({
        'Sales': 'sum',
        'Gross Profit': 'sum',
        'Units': 'sum',
        'Cost': 'sum'
    }).reset_index()

    product_margin['Gross_Margin'] = (
        product_margin['Gross Profit'] / product_margin['Sales']
    ) * 100

    product_margin['Profit_per_Unit'] = (
        product_margin['Gross Profit'] / product_margin['Units']
    )

    product_margin['Total_Profit_Contribution'] = product_margin['Gross Profit']

    product_margin['Cost_Ratio'] = (
        product_margin['Cost'] / product_margin['Sales']
    )

    product_margin['Rank_Gross_Profit'] = product_margin['Gross Profit'].rank(ascending=False)
    product_margin['Rank_Gross_Margin'] = product_margin['Gross_Margin'].rank(ascending=False)

    return product_margin


def classify_products(product_margin: pd.DataFrame) -> pd.DataFrame:
    """
    Tags each product as High/Low profit + margin segment.
    """
    avg_profit = product_margin['Gross Profit'].mean()
    avg_margin = product_margin['Gross_Margin'].mean()
    avg_sales  = product_margin['Sales'].mean()

    def segment(row):
        if row['Gross Profit'] > avg_profit and row['Gross_Margin'] > avg_margin:
            return 'High Profit / High Margin'
        elif row['Sales'] > avg_sales and row['Gross_Margin'] < avg_margin:
            return 'High Sales / Low Margin'
        elif row['Sales'] < avg_sales and row['Gross Profit'] < avg_profit:
            return 'Low Sales / Low Profit'
        else:
            return 'Average'

    product_margin['Segment'] = product_margin.apply(segment, axis=1)
    return product_margin


def calculate_division_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates division-level metrics with margin classification.
    """
    df['Gross_Margin'] = df['Gross Profit'] / df['Sales']

    division_summary = df.groupby('Division').agg({
        'Sales': 'sum',
        'Gross Profit': 'sum',
        'Gross_Margin': 'mean'
    }).reset_index()

    division_summary['Gross_Margin_%'] = division_summary['Gross_Margin'] * 100
    division_summary['Profit_Ratio'] = (
        division_summary['Gross Profit'] / division_summary['Sales']
    )

    avg_margin = division_summary['Gross_Margin'].mean()
    avg_sales  = division_summary['Sales'].mean()

    def classify_division(row):
        if row['Gross_Margin'] >= avg_margin and row['Sales'] >= avg_sales:
            return 'Strong Financial Efficiency'
        elif row['Gross_Margin'] < avg_margin and row['Sales'] >= avg_sales:
            return 'Structural Margin Issues'
        else:
            return 'Underutilized / Low Scale'

    division_summary['Category'] = division_summary.apply(classify_division, axis=1)
    return division_summary


def calculate_pareto(df: pd.DataFrame) -> tuple:
    """
    Returns (sales_pareto_df, profit_pareto_df) with cumulative % columns.
    """
    product_summary = df.groupby('Product Name').agg({
        'Sales': 'sum',
        'Gross Profit': 'sum'
    }).reset_index()

    # Sales Pareto
    sales_pareto = product_summary.sort_values('Sales', ascending=False).copy()
    sales_pareto['Cum_Sales'] = sales_pareto['Sales'].cumsum()
    sales_pareto['Cum_Sales_%'] = sales_pareto['Cum_Sales'] / sales_pareto['Sales'].sum() * 100

    # Profit Pareto
    profit_pareto = product_summary.sort_values('Gross Profit', ascending=False).copy()
    profit_pareto['Cum_Profit'] = profit_pareto['Gross Profit'].cumsum()
    profit_pareto['Cum_Profit_%'] = profit_pareto['Cum_Profit'] / profit_pareto['Gross Profit'].sum() * 100

    return sales_pareto, profit_pareto


def calculate_cost_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Product-level cost structure analysis with action flags.
    """
    product_diag = df.groupby('Product Name').agg({
        'Sales': 'sum',
        'Cost': 'sum',
        'Gross Profit': 'sum',
        'Units': 'sum'
    }).reset_index()

    product_diag['Gross_Margin'] = product_diag['Gross Profit'] / product_diag['Sales']
    product_diag['Cost_Ratio']   = product_diag['Cost'] / product_diag['Sales']

    avg_margin     = product_diag['Gross_Margin'].mean()
    avg_cost_ratio = product_diag['Cost_Ratio'].mean()
    avg_sales      = product_diag['Sales'].mean()

    def action_flag(row):
        if row['Gross_Margin'] < avg_margin and row['Cost_Ratio'] > avg_cost_ratio:
            return 'Cost Reduction / Renegotiation'
        elif row['Sales'] > avg_sales and row['Gross_Margin'] < avg_margin:
            return 'Repricing Needed'
        elif row['Sales'] < avg_sales and row['Gross_Margin'] < avg_margin:
            return 'Discontinuation Review'
        else:
            return 'Healthy'

    product_diag['Action'] = product_diag.apply(action_flag, axis=1)
    return product_diag