import React from 'react'
import {Button, Select} from "antd";
import {PlayCircleOutlined} from "@ant-design/icons";
import {useSelector} from "react-redux";

const CameraPreview = () => {
  const error = useSelector((state) => state.demo.error)
  return (
    <div className="preview-container">
      <img className="stream-preview" src={error ? "https://user-images.githubusercontent.com/5244214/151023826-313848a4-435b-4228-987a-b5b2017668b2.png" : "/stream"}/>
      <div className="preview-controls">
        <Select defaultValue="depth">
          <Select.Option value="depth">Depth</Select.Option>
          <Select.Option value="color">Color</Select.Option>
          <Select.Option value="disparity" disabled>
            Disparity
          </Select.Option>
          <Select.Option value="left">left</Select.Option>
        </Select>
        <Select defaultValue="184212512512CF">
          <Select.Option value="184212512512CF">184212512512CF</Select.Option>
        </Select>
        <Button danger type="primary" shape="round" icon={<PlayCircleOutlined/>} size="large">
          Reload
        </Button>
      </div>
    </div>
  );
};

export default CameraPreview;