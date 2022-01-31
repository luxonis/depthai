import {Button, Col, Input, Row, Select, Slider, Switch, Typography, Divider} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";
import {useSelector} from "react-redux";

function DepthProperties() {
  const config = useSelector((state) => state.demo.config).depth || {}

  return (
    <>
      <div className="title-section">
        <Typography.Title level={2}>Depth Properties</Typography.Title>
        <div><Switch checked={config.enabled}/> <span>Enabled</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Typography.Title level={3}>Median filtering</Typography.Title>
        <Select size="large" value={config.median.current}>
          {
            config.median.available.map(val => (
              <Select.Option value={val.startsWith("KERNEL_") ? +(val.replace("KERNEL_", "").split("x")[0]) : 0}>{val}</Select.Option>
            ))
          }
        </Select>
        <Divider />
        <div className="switchable-option"><Switch checked={config.subpixel}/> <span>Subpixel</span></div>
        <div className="switchable-option"><Switch checked={config.lrc}/> <span>Left Right Check</span></div>
        <div className="switchable-option"><Switch checked={config.extended}/> <span>Extended Disparity</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Typography.Title level={3}>Confidence Threshold</Typography.Title>
        <Slider marks={{0: '0', 255: '255'}} min={0} max={255} value={config.confidence}/>
        <Typography.Title level={3}>Bilateral Sigma</Typography.Title>
        <Slider marks={{0: '0', 250: '250'}} min={0} max={250} value={config.sigma}/>
        <Typography.Title level={3}>LRC Threshold</Typography.Title>
        <Slider marks={{0: '0', 10: '10'}} min={0} max={10} value={config.lrcThreshold}/>
        <Typography.Title level={3}>Depth Range (m)</Typography.Title>
        <Row className="input-box">
          <Col flex={2}>
            <Input addonBefore="min" value={config.range.min}/>
          </Col>
          <Col flex={2}>
            <Input addonBefore="max" value={config.range.max}/>
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
