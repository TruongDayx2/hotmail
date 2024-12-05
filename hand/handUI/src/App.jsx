import  { useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictedAngle, setPredictedAngle] = useState(null);
  const [error, setError] = useState(null);

  // Xử lý khi người dùng chọn file
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPredictedAngle(null); // Reset kết quả trước đó
    setError(null); // Reset lỗi trước đó
  };

  // Gửi file đến API và nhận kết quả
  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select an image file.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("http://localhost:8080/predict/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction.");
      }

      const data = await response.json();
      setPredictedAngle(data.angle);
    } catch (err) {
      setError("Error: " + err.message);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Hand Angle Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit">Predict</button>
      </form>

      {predictedAngle !== null && (
        <div>
          <h2>Predicted Angle: {predictedAngle.toFixed(2)} degrees</h2>
        </div>
      )}

      {error && (
        <div>
          <h2 style={{ color: "red" }}>{error}</h2>
        </div>
      )}
    </div>
  );
}

export default App;
