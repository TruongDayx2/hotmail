import axios from "axios";
import { useEffect, useState } from "react";
import handImg from "../img/hand.png";
import VerticalLine from "../components/verticalLine";
import RadioButton from "../components/radio";


const DegreeData = [
  '0','22.5','45','67.5','90','112.5','135','157.5','180','202.5','225','247.5','270','292.5','315','337.5'
]



function ToolPhanHuong() {
  const [url, setUrl] = useState("");
  const [urlObj, setUrlObj] = useState("");
  const [imgs, setImgs] = useState({ images: [] }); // Initialize with default structure
  const [currentImg, setCurrentImg] = useState(null);
  const [nameObj, setNameObj] = useState({ folders: [] });
  const [currentNameObj, setCurrentNameObj] = useState(null);
  const [selectedValue, setSelectedValue] = useState(null);
  const [folderRemove, setFolderRemove] = useState(null);
  const [folderAdd, setFolderAdd] = useState(null);

  // console.log("nameObj", nameObj);
  const [error, setError] = useState(null);

  // Handle input URL change
  const handleUrlChange = (event) => {
    setUrl(event.target.value);
    setError(null); // Reset any existing error
  };
  const handleUrlObjChange = (event) => {
    setUrlObj(event.target.value);
    setError(null); // Reset any existing error
  };
  // Submit the folder path to the API
  // Function to handle image submission
  const handleImageSubmit = async (event) => {
    event.preventDefault();
    if (url === "") {
      setError("Please provide a folder path.");
      return;
    }

    try {
      const folderPath = "http://localhost:8000/phanLoai/list-images/";
      const response = await axios.post(
        folderPath,
        { folder_path: url },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      setImgs(response.data); // Update image list
      setError(null); // Reset error
    } catch (err) {
      setImgs({ images: [] }); // Reset images on error
      setError("Error: " + (err.response?.data?.detail || err.message));
    }
  };

  // Function to handle object submission
  const handleObjectSubmit = async (event) => {
    event.preventDefault();
    if (urlObj === "") {
      setError("Please provide a folder path.");
      return;
    }

    try {
      const folderPath = "http://localhost:8000/phanLoai/list-objects/";
      const response = await axios.post(
        folderPath,
        { folder_path: urlObj },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      setNameObj(response.data); // Update object list
      setError(null); // Reset error
    } catch (err) {
      setNameObj({ folders: [] }); // Reset objects on error
      setError("Error: " + (err.response?.data?.detail || err.message));
    }
  };

  useEffect(() => {
    // khi báo hàm lấy data
    if (imgs.images.length > 0) {
      setCurrentImg(imgs.images[0]);
    } else {
      setCurrentImg(null);
    }
  }, [imgs]);

  const finalOK = async (e) => {
    e.preventDefault();
    if(currentNameObj === null || currentNameObj === ""){
      setError("Chọn tên đối tượng");
      return;
    }
    if(selectedValue === null){
      setError("Chọn Độ");
      return;
    }
    if(currentImg === null){
      setError("Chọn Hình");
      return;
    }
    if(folderRemove === null){
      setError("Chọn Thư mục chứa ảnh");
      return;
    }
    if(folderAdd === null){
      setError("Chọn Thư mục chứa đối tượng");
      return;
    }
    console.log(currentImg,selectedValue,currentNameObj,folderRemove,folderAdd)
    try {
      const folderPath = "http://localhost:8000/phanLoai/move-image/";
      const response = await axios.post(
        folderPath,
        { folder_path: folderRemove, image_name: currentImg,object_name: currentNameObj,angle_degree: selectedValue, folder_add:folderAdd},
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      console.log(response)
      setImgs((prevImgs) => ({
        images: prevImgs.images.slice(1), // Tạo mảng mới từ phần tử thứ 2 trở đi
      }));
      setSelectedValue(null); 
      setError(null); // Reset error
    } catch (err) {
      
      setError("Error: " + (err.response?.data?.detail || err.message));
    }
  };

  const handleChangeNameObj = (e) => {
    e.preventDefault();
    setCurrentNameObj(e.target.value);
  };

  const handleFolderRemove = (e,type) => {
    e.preventDefault();
    if(type === "img"){
      setFolderRemove(e.target.value);
    }else{
      setFolderAdd(e.target.value);
    }
  };

  const handleChoiceFolder =(e)=>{
    e.preventDefault();
    setCurrentNameObj(e.target.value);
  }

  const handleRadioChange = (value) => {
    setSelectedValue(value);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", alignItems: "center" }}>
      <h1>Phân hướng Đối tượng</h1>
      <div style={{ display: "flex", width: "100%", gap: "10px" }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <h3>Hình ảnh</h3>
          <form
            onSubmit={handleImageSubmit}
            style={{ display: "flex", justifyContent: "center", gap: "10px", marginBottom: "10px" }}
          >
            <label htmlFor="folderPath">Folder path:</label>
            <input id="folderPath" type="text" onChange={handleUrlChange} value={url} />
            <button style={{ backgroundColor: "#00FFCC" }} type="submit">
              Get
            </button>
          </form>

          {currentImg && (
            <>
              <img
                src={`http://localhost:8000/phanLoai/images/${currentImg}?folder_path=${url}`}
                alt={currentImg}
                style={{
                  width: "auto",
                  height: "auto",
                }}
              />
              <p>{currentImg}</p>
            </>
          )}
        </div>
        <VerticalLine />
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <h3>Đối tượng</h3>
          <form
            onSubmit={handleObjectSubmit}
            style={{ display: "flex", justifyContent: "center", gap: "10px", marginBottom: "10px" }}
          >
            <label htmlFor="folderPath">Folder path:</label>
            <input id="folderPath" type="text" onChange={handleUrlObjChange} value={urlObj} />
            <button style={{ backgroundColor: "#00FFCC" }} type="submit">
              Get
            </button>
          </form>
          {nameObj.folders.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
              {nameObj.folders.map((item, index) => (
                <button
                  key={index}
                  value={item}
                  onClick={handleChoiceFolder}
                  style={{
                    backgroundColor: "green",
                    color: "white",
                    padding: "8px 12px",
                    borderRadius: "4px",
                    border: "none",
                    cursor: "pointer",
                    flex: "0 0 calc(25% - 10px)",
                    boxSizing: "border-box",
                  }}
                >
                  {item}
                </button>
              ))}
            </div>
          )}

          <p>Tên đối tượng</p>
          <input value={currentNameObj} onChange={handleChangeNameObj} placeholder="duck" />
        </div>
        <VerticalLine />
        <div>
          <h3>Độ: &emsp;   {selectedValue}</h3>
          {DegreeData && (
           <div
           style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)", // 4 cột mỗi hàng
            gap: "10px", // Khoảng cách giữa các phần tử
            width: "500px", // Giới hạn độ rộng
           }}
         >
           {DegreeData.map((item, index) => (
             <RadioButton
               key={index}
               value={item}
               selectedValue={selectedValue}
               onChange={handleRadioChange}
               
             />
           ))}
           
         </div>
     
          )}
        
        </div>
        <VerticalLine />
        <div>
          <h3>Thư mục chứa ảnh</h3>
          <input type="text" onChange={e=>handleFolderRemove(e,"img")} value={folderRemove}/>
          <h3>Thư mục chứa đối tượng</h3>
          <input type="text" onChange={e=>handleFolderRemove(e,"folder")} value={folderAdd}/>
        </div>
      </div>
      
      {error && <p style={{ color: "red" }}>{error}</p>}
      <button
        style={{ backgroundColor: "#00FFCC", width: "20%", marginTop: "10px" }}
        onClick={finalOK}
      >
        OK
      </button>
      <div
        style={{
          marginTop: "10px",
          textAlign: "center",
        }}
      >
        <h3>Hình ảnh hỗ trợ</h3>
        <img src={handImg} alt="Hỗ trợ" style={{ width: "80%", height: "auto" }} />
      </div>
    </div>
  );
}

export default ToolPhanHuong;
