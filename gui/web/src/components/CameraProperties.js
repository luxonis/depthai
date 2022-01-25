import {Button, Col, Input, Row, Select, Slider, Switch, Typography} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";

function CameraProperties() {
  return (
    <>
      <div className="title-section">
        <Typography.Title level={2}>Camera Properties</Typography.Title>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Row>
          <Col span={12}>
            <Typography.Title level={3}>Color</Typography.Title>
            <Input addonBefore="FPS" defaultValue="30"/>
            <span>Resolution</span>
            <Select defaultValue="THE_1080_P">
              <Select.Option value="THE_1080_P">THE_1080_P</Select.Option>
              <Select.Option value="THE_4K">THE_4K</Select.Option>
              <Select.Option value="THE_13MP">THE_13MP</Select.Option>
            </Select>
          </Col>
          <Col span={12}>
            <Typography.Title level={3}>Left + Right</Typography.Title>
            <Input addonBefore="FPS" defaultValue="30"/>
            <span>Resolution</span>
            <Select defaultValue="THE_1080_P">
              <Select.Option value="THE_1080_P">THE_1080_P</Select.Option>
              <Select.Option value="THE_4K">THE_4K</Select.Option>
              <Select.Option value="THE_13MP">THE_13MP</Select.Option>
            </Select>
          </Col>
        </Row>
        <div className="switchable-option"><Switch/> <span>Enable sync</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Row>
          <Col span={12}>
            <Input addonBefore="ISO"/>
            <Input addonBefore="Exposure"/>
            <div>
              <span>Saturation</span>
              <Slider marks={{0: '0', 10: '10'}} min={0} max={10} defaultValue={5}/>
            </div>
            <div>
              <span>Contrast</span>
              <Slider marks={{0: '0', 10: '10'}} min={0} max={10} defaultValue={5}/>
            </div>
            <div>
              <span>Brightness</span>
              <Slider marks={{0: '0', 10: '10'}} min={0} max={10} defaultValue={5}/>
            </div>
            <div>
              <span>Sharpness</span>
              <Slider marks={{0: '0', 4: '4'}} min={0} max={4} defaultValue={0}/>
            </div>
          </Col>
          <Col span={12}>
            <Input addonBefore="ISO"/>
            <Input addonBefore="Exposure"/>
            <div>
              <span>Saturation</span>
              <Slider marks={{0: '0', 10: '10'}} min={0} max={10} defaultValue={5}/>
            </div>
            <div>
              <span>Contrast</span>
              <Slider marks={{0: '0', 10: '10'}} min={0} max={10} defaultValue={5}/>
            </div>
            <div>
              <span>Brightness</span>
              <Slider marks={{0: '0', 10: '10'}} min={0} max={10} defaultValue={5}/>
            </div>
            <div>
              <span>Sharpness</span>
              <Slider marks={{0: '0', 4: '4'}} min={0} max={4} defaultValue={0}/>
            </div>
          </Col>
        </Row>
      </div>
      <Button className="restart-button" type="primary" block size="large">
        Apply and Restart
      </Button>
    </>
  );
}

export default CameraProperties;
