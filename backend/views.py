from django.shortcuts import render
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from django.views import View
import os
from django.conf import settings
import pandas as pd  # assuming you're using pandas to load the CSV

class home(View):
     def get(self, request):
         return render(request, "index.html", {"prediction": ""})


     def post(self, request): 
            file_path = os.path.join(settings.BASE_DIR,'frontend', 'data', 'data.csv')  # make sure this line is indented with 4 spaces or a single tab
            df = pd.read_csv(file_path)
            d1 = float(request.POST.get("one"))
            d2 = float(request.POST.get("two"))
            d3 = float(request.POST.get("three"))
            d4 = float(request.POST.get("four"))
            d5 = float(request.POST.get("five"))
            d6 = request.POST.get("six")
            d7 = float(request.POST.get("seven"))
            d8 = float(request.POST.get("eight"))

            d9 = float(request.POST.get("nine"))
            d10 = float(request.POST.get("ten"))
            d6 = 1 if d6.lower() == "yes" else 0

# Load dataset
           



# Handle missing values (numeric only)
            df = df.select_dtypes(include=[np.number])
            df = df.fillna(df.mean())

# Feature and target
            X = df.drop(['price', 'statezip', 'street'], axis=1, errors='ignore')
            y = df['price']

# Save column names
            feature_names = X.columns.tolist()

# Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

# Train model
            user_input_df = pd.DataFrame([[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]], columns=feature_names)
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)

# Prepare user input with same feature names
           

# Scale user input
            user_input_scaled = scaler.transform(user_input_df)

# Predict
            prediction = lr.predict(user_input_scaled)
            p= prediction[0]
            print(p)
            return render(request,'index.html',{"prediction":p})

