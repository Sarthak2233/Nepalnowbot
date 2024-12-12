 import React, { useEffect, useState } from "react";
 import { Link } from "react-router-dom";

function Front() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8002/") // FastAPI backend URL
      .then((response) => response.json())
      .then((data) => setData(data));
  }, []);

  return (
    <div className='landingPage'>
      <h1>FastAPI and React Integration</h1>
      {data ? (
        <div>
          <p>{data.name}</p>
          <p>{data.response}</p>
          <p>{data.message}</p>
        </div>
      ) : (
        <p>Loading...</p>
      )}
      <a href="/">Go to ChatPage</a>
    </div>
  );
}

export default Front;
