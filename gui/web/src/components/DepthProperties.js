import {Button, Col, Input, Row, Select, Slider, Switch, Typography} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";

function DepthProperties() {
  return (
    <>
      <div className="title-section">
        <Typography.Title level={2}>Depth Properties</Typography.Title>
        <div><Switch/> <span>Enabled</span><Switch/> <span>Use Disparity</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Typography.Title level={3}>Median filtering</Typography.Title>
        <Select size="large" defaultValue="7x7">
          <Select.Option value="7x7">Kernel 7x7</Select.Option>
          <Select.Option value="5x5">Kernel 5x5</Select.Option>
          <Select.Option value="3x3">Kernel 3x3</Select.Option>
          <Select.Option value="None">No filtering</Select.Option>
        </Select>
        <div className="switchable-option"><Switch/> <span>Subpixel</span></div>
        <div className="switchable-option"><Switch/> <span>Left Right Check</span></div>
        <div className="switchable-option"><Switch/> <span>Extended Disparity</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Typography.Title level={3}>Confidence Threshold</Typography.Title>
        <Slider marks={{0: '0', 255: '255'}} min={0} max={255} defaultValue={240}/>
        <Typography.Title level={3}>Bilateral Sigma</Typography.Title>
        <Slider marks={{0: '0', 250: '250'}} min={0} max={250} defaultValue={0}/>
        <Typography.Title level={3}>LRC Threshold</Typography.Title>
        <Slider marks={{0: '0', 10: '10'}} min={0} max={10} defaultValue={0}/>
        <Typography.Title level={3}>Depth Range (m)</Typography.Title>
        <Row>
          <Col flex={2}>
            <Input addonBefore="min" defaultValue="0"/>
          </Col>
          <Col flex={2}>
            <Input addonBefore="max" defaultValue="10"/>
          </Col>
        </Row>
      </div>
      <Button className="restart-button" type="primary" block size="large">
        Apply and Restart
      </Button>
    </>
  );
}

export default DepthProperties;
