import {Button, Col, Input, Row, Select, Slider, Switch, Typography} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";

function MiscProperties() {
  return (
    <>
      <Row>
        <Col span={12}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Recording</Typography.Title>
            <div className="switchable-option"><Switch/> <span>Color</span> <Input addonBefore="FPS" defaultValue="30"/></div>
            <div className="switchable-option"><Switch/> <span>Left</span> <Input addonBefore="FPS" defaultValue="30"/></div>
            <div className="switchable-option"><Switch/> <span>Right</span> <Input addonBefore="FPS" defaultValue="30"/></div>
            <Input addonBefore="Destination" type="file"/>
          </div>
        </Col>
        <Col span={12}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Demo options</Typography.Title>
            <div className="switchable-option"><Switch/> <span>Send anonymous usage data</span></div>
          </div>
        </Col>
      </Row>
      <Row>
        <Col span={12}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Recording</Typography.Title>
            <div className="switchable-option"><Switch/> <span>Temperature</span></div>
            <div className="switchable-option"><Switch/> <span>CPU</span></div>
            <div className="switchable-option"><Switch/> <span>Memory</span></div>
            <Input addonBefore="Destination" type="file"/>
          </div>
        </Col>
        <Col span={12}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Applications</Typography.Title>
            <div><span>UVC Mode (Webcam) <Button type="primary">Run</Button></span></div>
          </div>
        </Col>
      </Row>
      <Button type="primary" block size="large">
        Apply and Restart
      </Button>
    </>
  );
}

export default MiscProperties;
