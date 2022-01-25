import {Button, Col, Input, Row, Select, Slider, Switch, Typography} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";

function AIProperties() {
  return (
    <>
      <div className="title-section">
        <Typography.Title level={2}>AI Properties</Typography.Title>
        <div><Switch/> <span>Enabled</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Typography.Title level={3}>Neural network</Typography.Title>
        <Select size="large" defaultValue="mobilenet-ssd">
          <Select.Option value="mobilenet-ssd">mobilenet-ssd</Select.Option>
          <Select.Option value="face-detection-retail-0004">face-detection-retail-0004</Select.Option>
          <Select.Option value="openpose2">openpose2</Select.Option>
          <Select.Option value="vehicle-detection-adas-0001">vehicle-detection-adas-0001</Select.Option>
        </Select>
        <div className="switchable-option"><Switch/> <span>Full FOV input</span></div>
        <div>
          <Typography.Title level={3}>SHAVEs</Typography.Title>
          <Slider marks={{0: '0', 12: '12'}} min={0} max={12} defaultValue={6}/>
        </div>
        <div>
          <span>Model source</span>
          <Select defaultValue="color">
            <Select.Option value="color">color</Select.Option>
            <Select.Option value="left">left</Select.Option>
            <Select.Option value="right">right</Select.Option>
          </Select>
        </div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <div>
          <span>OpenVINO version</span>
          <Select defaultValue="VERSION_2021_4">
            <Select.Option value="VERSION_2021_4">VERSION_2021_4</Select.Option>
            <Select.Option value="VERSION_2021_4">VERSION_2021_3</Select.Option>
            <Select.Option value="VERSION_2021_4">VERSION_2021_2</Select.Option>
          </Select>
        </div>
        <div>
          <span>Label to count</span>
          <Select defaultValue="color">
            <Select.Option value="color">color</Select.Option>
            <Select.Option value="left">left</Select.Option>
            <Select.Option value="right">right</Select.Option>
          </Select>
        </div>
        <div className="switchable-option"><Switch/> <span>Spatial Bounding Boxes (SBB)</span></div>
        <div>
          <span>SBB Factor</span>
          <Slider marks={{0: '0', 1: '1'}} step={0.1} min={0} max={1} defaultValue={0.3}/>
        </div>
      </div>
      <Button type="primary" block size="large">
        Apply and Restart
      </Button>
    </>
  );
}

export default AIProperties;
