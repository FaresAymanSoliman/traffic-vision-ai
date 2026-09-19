import { useEffect, useState } from "react";
import "./App.css";

type HealthResponse = {
  status: "healthy";
  service: string;
  environment: string;
};

type RequestState =
  | { status: "loading" }
  | { status: "success"; data: HealthResponse }
  | { status: "error"; message: string };

function App() {
  const [requestState, setRequestState] = useState<RequestState>({
    status: "loading",
  });

  useEffect(() => {
    const loadHealth = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/health");

        if (!response.ok) {
          throw new Error(`API request failed with status ${response.status}`);
        }

        const data = (await response.json()) as HealthResponse;

        setRequestState({
          status: "success",
          data,
        });
      } catch (error) {
        setRequestState({
          status: "error",
          message:
            error instanceof Error
              ? error.message
              : "An unexpected error occurred.",
        });
      }
    };

    void loadHealth();
  }, []);

  return (
    <main className="page">
      <section className="card">
        <p className="eyebrow">TrafficVision AI</p>

        <h1>System Foundation</h1>

        <p className="description">
          React is checking the FastAPI backend connection.
        </p>

        {requestState.status === "loading" && (
          <div className="status status-loading">Checking API...</div>
        )}

        {requestState.status === "error" && (
          <div className="status status-error">
            <strong>Backend unavailable</strong>
            <span>{requestState.message}</span>
          </div>
        )}

        {requestState.status === "success" && (
          <div className="status status-success">
            <strong>Backend connected</strong>

            <dl>
              <div>
                <dt>Status</dt>
                <dd>{requestState.data.status}</dd>
              </div>

              <div>
                <dt>Service</dt>
                <dd>{requestState.data.service}</dd>
              </div>

              <div>
                <dt>Environment</dt>
                <dd>{requestState.data.environment}</dd>
              </div>
            </dl>
          </div>
        )}
      </section>
    </main>
  );
}

export default App;