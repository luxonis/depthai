import {Button, Select, Slider, Switch, Typography} from "antd";
import {QuestionCircleOutlined} from "@ant-design/icons";
import {useSelector, useDispatch} from "react-redux";
import {updateConfig} from "../store";

function AIProperties() {
  const config = useSelector((state) => state.demo.config).ai || {}
  const dispatch = useDispatch()

  const update = data => dispatch(updateConfig({ai: data}))
  return (
    <>
      <div className="title-section">
        <Typography.Title level={2}>AI Properties</Typography.Title>
        <div><Switch onChange={enabled => update({enabled})} checked={config.enabled}/> <span>{config.enabled ? "Enabled" : "Disabled"}</span></div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <Typography.Title level={3}>Neural network</Typography.Title>
        <Select onChange={current => update({model: {current}})} disabled={!config.enabled} size="large" value={config.model ? config.model.current : null}>
          {
            config.model && config.model.available.map(
              network => <Select.Option key={network} value={network}>{network}</Select.Option>
            )
          }
        </Select>
        <div className="switchable-option"><Switch disabled={!config.enabled} onChange={fullFov => update({fullFov})} checked={config.fullFov}/> <span>Full FOV input</span></div>
        <div>
          <Typography.Title level={3}>SHAVEs</Typography.Title>
          <Slider disabled={!config.enabled} tooltipVisible={false} onChange={shaves => update({shaves})} marks={{0: '0', [config.shaves]: '' + config.shaves, 12: '12'}} min={0} max={12} value={config.shaves}/>
        </div>
        <div>
          <span>Model source</span>
          <Select disabled={!config.enabled} onChange={current => update({source: {current}})} value={config.source ? config.source.current : null}>
            {
              config.source && config.source.available.map(
                source => <Select.Option key={source} value={source}>{source}</Select.Option>
              )
            }
          </Select>
        </div>
      </div>
      <div className="options-section">
        <a href="#" className="info-indicator"><QuestionCircleOutlined/></a>
        <div className="single-option">
          <span>OpenVINO version</span>
          <Select disabled={!config.enabled} onChange={current => update({ovVersion: {current}})} value={config.ovVersion ? config.ovVersion.current : null}>
            {
              config.ovVersion && config.ovVersion.available.map(
                ovVersion => <Select.Option key={ovVersion} value={ovVersion}>{ovVersion}</Select.Option>
              )
            }
          </Select>
        </div>
        <div className="single-option">
          <span>Label to count</span>
          <Select disabled={!config.enabled} onChange={current => update({label: {current}})} value={config.label && config.label.current ? config.label.current : "None"}>
            <Select.Option value="null">None</Select.Option>
            {
              config.label && config.label.available.map(
                label => <Select.Option key={label} value={label}>{label}</Select.Option>
              )
            }
          </Select>
        </div>
        <div className="switchable-option"><Switch disabled={!config.enabled} onChange={sbb => update({sbb})} checked={config.sbb}/> <span>Spatial Bounding Boxes (SBB)</span></div>
        <div>
          <span>SBB Factor</span>
          <Slider disabled={!config.enabled} tooltipVisible={false} onChange={sbbFactor => update({sbbFactor})} marks={{0: '0', [config.sbbFactor]: '' + config.sbbFactor, 1: '1'}} step={0.1} min={0} max={1} value={config.sbbFactor}/>
        </div>
      </div>
    </>
  );
}

export default AIProperties;
