import {Button, Col, Input, Row, Select, Slider, Switch, Typography, Divider} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";
import {useDispatch, useSelector} from "react-redux";
import {sendConfig, sendDynamicConfig, updateConfig, updateDynamicConfig} from "../store";

function DepthProperties() {
  const config = useSelector((state) => state.demo.config).depth || {}
  const dispatch = useDispatch()

  const update = data => dispatch(updateConfig({depth: data}))
  const updateDynamic = data => {
    dispatch(updateDynamicConfig({depth: data}))
    dispatch(sendDynamicConfig())
  }

  return (
    <>
      <div className="title-section">
        <Typography.Title level={2}>Depth Properties</Typography.Title>
        <div><Switch checked={config.enabled} onChange={enabled => update({enabled})}/> <span>Enabled</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Typography.Title level={3}>Median filtering</Typography.Title>
        <Select onChange={current => updateDynamic({median: {current}})} size="large" value={config.median.current}>
          {
            config.median.available.map(val => (
              <Select.Option value={val.startsWith("KERNEL_") ? +(val.replace("KERNEL_", "").split("x")[0]) : 0}>{val}</Select.Option>
            ))
          }
        </Select>
        <Divider />
        <div className="switchable-option"><Switch onChange={subpixel => update({subpixel})} checked={config.subpixel}/> <span>Subpixel</span></div>
        <div className="switchable-option"><Switch onChange={lrc => update({lrc})} checked={config.lrc}/> <span>Left Right Check</span></div>
        <div className="switchable-option"><Switch onChange={extended => update({extended})} checked={config.extended}/> <span>Extended Disparity</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Typography.Title level={3}>Confidence Threshold</Typography.Title>
        <Slider onChange={confidence => updateDynamic({confidence})} marks={{0: '0', 255: '255'}} min={0} max={255} value={config.confidence}/>
        <Typography.Title level={3}>Bilateral Sigma</Typography.Title>
        <Slider onChange={sigma => updateDynamic({sigma})} marks={{0: '0', 250: '250'}} min={0} max={250} value={config.sigma}/>
        <Typography.Title level={3}>LRC Threshold</Typography.Title>
        <Slider onChange={lrcThreshold => updateDynamic({lrcThreshold})} marks={{0: '0', 10: '10'}} min={0} max={10} value={config.lrcThreshold}/>
        <Typography.Title level={3}>Depth Range (m)</Typography.Title>
        <Row className="input-box">
          <Col flex={2}>
            <Input addonBefore="min" onChange={min => updateDynamic({range: {min}})} value={config.range.min}/>
          </Col>
          <Col flex={2}>
            <Input addonBefore="max" onChange={max => updateDynamic({range: {max}})} value={config.range.max}/>
          </Col>
        </Row>
      </div>
    </>
  );
}

export default DepthProperties;
