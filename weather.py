import requests

# Your API Key and Location
api_key = "52d129c124aaa073f4208583f3299b19"
location = "Sangli"

def get_weather_info(location:str , api_key:str):
    """
    This function fetches weather data from OWM
    """
    
    # Link
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

    response = requests.get(url)
    return response.json()

def give_advice(weather_data):
    # weather data

    temp = weather_data['main']['temp']
    humidity  = weather_data['main']['humidity']
    rain = weather_data.get('rain',{}).get('6h',0)

    advice = []

    print(f"location : {location}")
    print('\n')
    print(f"temperature : {temp} .")
    print(f"humidity :  {humidity} .")
    print(f"Rain in last 6 hours : {rain} mm.")
    
    
    if temp < 35 and humidity in range(50,70):
        text = "This is good weather for crops . Good time water the crops."
        advice.append(text)
    
    elif temp < 35:
        text = "Its too Hot today. Water in the morinig or evening to keep plants cool."
        advice.append(text)
    
    elif temp in range(25,31) and humidity in range(60,80):
        text = "This is perfect time for planting or adding Fertilizers as crops take it well."
        advice.append(text)

    elif humidity >= 80:
        text = "Too much moisture can cause plant disease so don't overwater."
        advice.append(text)

    else:
        text = "Do other farming activities"
        advice.append(text)

    return advice


if __name__ == '__main__':

    # set a defualt location
    # ask user for location using input function
    # Use other import or function that gives user location

    #location = "Sangli"
    location = input("Enter Your Location : ")
    #location = fetch_location_function()
    weather_data = get_weather_info(location)
    advice = give_advice(weather_data)

    print("weather Based Farming Advice :")
    
    for line in advice:
        print(' - '+ line)
        print('\n')
