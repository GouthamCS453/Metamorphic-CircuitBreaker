# React Frontend

This folder contains a separate React/Vite dashboard for the additive fallback mechanism.

It does not replace the existing Streamlit application.

## Start

From this directory:

```bat
npm install
npm run dev
```

Vite normally starts the dashboard at:

```text
http://localhost:5173
```

The dashboard communicates with the FastAPI service at:

```text
http://127.0.0.1:8000
```

Start the backend first using the instructions in `backend/README.md`.

The dashboard can:

- upload a traffic-sign image;
- run the existing Metamorphic Circuit Breaker;
- display CLOSED, HALF_OPEN, or OPEN;
- display the fallback action;
- simulate driver takeover acknowledgement;
- display the driver-present/absent mode.

`SAFE_PULL_OVER` is a simulated event and does not control a physical vehicle.
