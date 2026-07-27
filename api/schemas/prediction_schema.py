from pydantic import BaseModel


class CustomerInput(BaseModel):
    Recency: float
    Frequency: float
    Monetary_Total: float
    Monetary_Average: float
    Total_Quantity: float
    Average_Basket_Size: float
    Average_Order_Value: float
    Unique_Products: float
    Average_Purchase_Gap: float
    Customer_Lifespan_Days: float
    Return_Rate: float
    Price_Sensitivity: float
    Predicted_CLV: float
    Engagement_Score: float
    Country: str