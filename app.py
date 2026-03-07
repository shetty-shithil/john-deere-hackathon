from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/health_check")
async def health_check():
    return {"status": "Health ok"}


@app.get("/location_info")
async def location_info():
    return {"location": "New York", "temperature": "25°C"}



class LocationRequest(BaseModel):
    lat: float
    lon: float

@app.post("/get_field_conditions")
async def get_field_conditions(location: LocationRequest):

    """
    Get field conditions based on latitude and longitude.
    Body: {"lat": 28.6139, "lon": 77.2090}
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "current_weather": True,
        "hourly": [
                    "temperature_2m",
                    "relativehumidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "rain",
                    "snowfall",
                    "uv_index",
                    "cloudcover",
                    "winddirection_10m",
                    "soil_temperature_0cm",
                    "soil_moisture_0_1cm",
                    "evapotranspiration"
                ]    
            }
    timeout = httpx.Timeout(
        connect=10.0,   # connection timeout
        read=30.0,      # read timeout
        write=10.0,     # write timeout
        pool=10.0       # pool timeout
        )
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Weather service unavailable")
        data = response.json()
        current = data["current_weather"]
        hourly = data["hourly"]
        # print(f"Received weather data: {data}")
        return {
            "latitude": location.lat,
            "longitude": location.lon,
            
            "temperature_c": current["temperature"],
            "windspeed_kmh": current["windspeed"],
            "condition_code": current["weathercode"],
            "is_day": bool(current["is_day"]),
            
            "humidity_%": hourly["relativehumidity_2m"][0],
            "feels_like_c": hourly["apparent_temperature"][0],
            "precipitation_mm": hourly["precipitation"][0],
            "rain_mm": hourly["rain"][0],
            "snowfall_cm": hourly["snowfall"][0],
            "uv_index": hourly["uv_index"][0],
            "cloudcover_%": hourly["cloudcover"][0],
            "windirection_deg": hourly["winddirection_10m"][0],
            
            "soil_temperature_c": hourly["soil_temperature_0cm"][0],
            "soil_moisture_%": hourly["soil_moisture_0_1cm"][0],
            "evapotranspiration_mm": hourly["evapotranspiration"][0],
        }
    


class ChatRequest(BaseModel):
    text_message: str
    image_base64: str
    lat: float
    lon: float



async def orchestrate(request: ChatRequest):
    results = {}

    results["field_conditions"]  = await get_field_conditions(LocationRequest(lat=request.lat, lon=request.lon))
    # results["image"]    = await process_image(request.image_base64)
    # results["message"]  = await process_message(request.text_message)

    return results


@app.post("/enter_chat")
async def enter_chat(request: ChatRequest):
    result = await orchestrate(request)
    return {
        "status": "success",
        "data": result
    }    