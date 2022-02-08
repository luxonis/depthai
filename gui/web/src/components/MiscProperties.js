import {Button, Col, Input, Row, Switch, Typography} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";
import {useDispatch, useSelector} from "react-redux";
import {updateConfig} from "../store";

function MiscProperties() {
  const config = useSelector((state) => state.demo.config).misc || {}
  const dispatch = useDispatch()

  const update = data => dispatch(updateConfig({misc: data}))

  return (
    <>
      <Row className="input-box">
        <Col span={24}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Recording</Typography.Title>
            <div className="switchable-option">
              <Switch onChange={enabled => update({recording: {color: {enabled}}})} checked={config.recording.color.enabled}/>
              <span>Color</span>
              <Input className="switchable-input" type="number" addonBefore="FPS" value={config.recording.color.fps} onChange={event => update({recording: {color: {fps: event.target.value}}})}/>
            </div>
            <div className="switchable-option">
              <Switch onChange={enabled => update({recording: {left: {enabled}}})} checked={!!config.recording.left.enabled}/>
              <span>Left</span>
              <Input className="switchable-input" addonBefore="FPS" value={config.recording.left.fps} onChange={event => update({recording: {left: {fps: event.target.value}}})}/>
            </div>
            <div className="switchable-option">
              <Switch onChange={enabled => update({recording: {right: {enabled}}})} checked={!!config.recording.right.enabled}/>
              <span>Right</span>
              <Input className="switchable-input" addonBefore="FPS" value={config.recording.right.fps} onChange={event => update({recording: {right: {fps: event.target.value}}})}/>
            </div>
            <Input addonBefore="Destination" value={config.recording.dest} onChange={event => update({recording: {dest: event.target.value}})}/>
          </div>
        </Col>
        <Col span={24}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Demo options</Typography.Title>
            <div className="switchable-option"><Switch onChange={statistics => update({demo: {statistics}})} checked={config.demo.statistics}/> <span>Send anonymous usage data</span></div>
          </div>
        </Col>
      </Row>
      <Row>
        <Col span={24}>
          <div className="options-section">
            <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
            <Typography.Title level={3}>Raporting</Typography.Title>
            <div className="switchable-option">
              <Switch onChange={enabled => update({reporting: {enabled: config.reporting.enabled.filter(item => item !== "temp") + (enabled ? ["temp"] : [])}})} checked={_.includes(config.reporting.enabled, "temp")}/>
              <span>Temperature</span>
            </div>
            <div className="switchable-option">
              <Switch onChange={enabled => update({reporting: {enabled: config.reporting.enabled.filter(item => item !== "cpu") + (enabled ? ["cpu"] : [])}})} checked={_.includes(config.reporting.enabled, "cpu")}/>
              <span>CPU</span>
            </div>
            <div className="switchable-option">
              <Switch onChange={enabled => update({reporting: {enabled: config.reporting.enabled.filter(item => item !== "memory") + (enabled ? ["memory"] : [])}})} checked={_.includes(config.reporting.enabled, "memory")}/>
              <span>Memory</span>
            </div>
            <Input addonBefore="Destination" value={config.reporting.dest} onChange={event => update({reporting: {dest: event.target.value}})}/>
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
    </>
  );
}

export default MiscProperties;
