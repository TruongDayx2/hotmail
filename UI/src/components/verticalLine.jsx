
function VerticalLine() {
  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "200px" }}>
      <div
        style={{
          borderLeft: "2px solid black", // Đường viền dọc
          height: "100%", // Chiều cao
        }}
      ></div>
    </div>
  );
}

export default VerticalLine;