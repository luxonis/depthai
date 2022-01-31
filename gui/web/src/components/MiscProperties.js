import {Button, Col, Input, Row, Switch, Typography} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";
import {useSelector} from "react-redux";

function MiscProperties() {
  const config = useSelector((state) => state.demo.config).misc || {}
  return (
    <>
      <Row className="input-box">
        <Col span={24}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Recording</Typography.Title>
            <div className="switchable-option"><Switch checked={!!config.recording.color}/> <span>Color</span> <Input className="switchable-input" addonBefore="FPS" value={config.recording.color}/>
            </div>
            <div className="switchable-option"><Switch checked={!!config.recording.left}/> <span>Left</span> <Input className="switchable-input" addonBefore="FPS" value={config.recording.left}/>
            </div>
            <div className="switchable-option"><Switch checked={!!config.recording.right}/> <span>Right</span> <Input className="switchable-input" addonBefore="FPS" value={config.recording.right}/>
            </div>
            <Input addonBefore="Destination" type="file"/>
            <span>(Current: {config.recording.dest})</span>
          </div>
        </Col>
        <Col span={24}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Demo options</Typography.Title>
            <div className="switchable-option"><Switch checked={config.demo.statistics}/> <span>Send anonymous usage data</span></div>
          </div>
        </Col>
      </Row>
      <Row>
        <Col span={24}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Raporting</Typography.Title>
            <div className="switchable-option"><Switch checked={_.includes(config.reporting.enabled, "temp")}/> <span>Temperature</span></div>
            <div className="switchable-option"><Switch checked={_.includes(config.reporting.enabled, "cpu")}/> <span>CPU</span></div>
            <div className="switchable-option"><Switch checked={_.includes(config.reporting.enabled, "memory")}/> <span>Memory</span></div>
            <Input addonBefore="Destination" type="file"/>
            <span>(Current: {config.reporting.dest})</span>
          </div>
        </Col>
        <Col span={24}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Applications</Typography.Title>
            <div><span>UVC Mode (Webcam) <Button type="primary">Run</Button></span></div>
          </div>
        </Col>
      </Row>
      <Button className="restart-button" type="primary" block size="large">
        Apply and Restart
      </Button>
    </>
  );
}

export default MiscProperties;
