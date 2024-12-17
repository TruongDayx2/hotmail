import axios from "axios";
import { useState } from "react";

function TestHand() {
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
      const response = await axios.post("http://localhost:8000/hand/predict/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setPredictedAngle(response.data.angle);
    } catch (err) {
      setPredictedAngle(null);
      setError("Error: " + (err.response?.data?.detail || err.message));
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Dự đoán hướng tay</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button style={{ backgroundColor: "#00FFCC" }} type="submit">
          Predict
        </button>
      </form>

      {predictedAngle !== null && (
        <div>
          <h2>Predicted Angle: {predictedAngle.toFixed(2)} degrees</h2>
        </div>
      )}

      {error && predictedAngle === null && (
        <div>
          <h2 style={{ color: "red" }}>{error}</h2>
        </div>
      )}
    </div>
  );
}

export default TestHand;
