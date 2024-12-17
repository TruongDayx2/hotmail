
// eslint-disable-next-line react/prop-types
const RadioButton = ({ value, selectedValue, onChange }) => {
  const isSelected = selectedValue === value;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        cursor: "pointer",
      }}
      onClick={() => onChange(value)}
    >
      {/* Vòng tròn radio */}
      <div
        style={{
          width: "16px",
          height: "16px",
          borderRadius: "50%",
          border: "2px solid green",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginRight: "8px",
        }}
      >
        {isSelected && (
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor: "green",
            }}
          ></div>
        )}
      </div>
      {/* Label */}
      <span style={{ color: "green" }}>{value}</span>
    </div>
  );
};

export default RadioButton;
