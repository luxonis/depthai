import React from 'react'
import {Button, Select} from "antd";
import {PlayCircleOutlined} from "@ant-design/icons";

const CameraPreview = () => {
  return (
    <div className="preview-container">
      <img className="stream-preview" src="/stream"/>
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