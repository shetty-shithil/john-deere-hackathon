# Flask API Boilerplate

This is a basic FastAPI template.

## Setup

1. Install dependencies: `pip install -r requirements.txt`

2. Run the app with Uvicorn server:
   ```sh
   uvicorn app:app --reload
   ```

## Usage

The API has a root endpoint that returns a JSON message.

## API Endpoints

- **GET /** : Returns `{ "message": "Hello, World!" }`