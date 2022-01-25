import {Button, Card, Col, Input, Row, Select, Slider, Spin, Switch, Tabs, Typography} from "antd";
import {
  CameraOutlined,
  BorderOuterOutlined,
  ExperimentOutlined,
  PlayCircleOutlined,
  SlidersOutlined, QuestionCircleOutlined
} from "@ant-design/icons";
import AIProperties from "./components/AIProperties";
import CameraPreview from "./components/CameraPreview";
import {useDispatch, useSelector} from "react-redux";
import {fetchConfig} from "./store";
import {useEffect} from "react";

function App() {
  const fetched = useSelector((state) => state.demo.fetched)
  const dispatch = useDispatch()

  useEffect(() => {
    if(!fetched) {
      dispatch(fetchConfig())
    }
  }, [fetched])

  return (
    <div className="root-container">
      <Row align="middle">
        <Col flex={1}>
          {
            fetched
              ? <CameraPreview/>
              : <Spin tip="Loading..."/>
          }
        </Col>
        <Col flex={1}>
          <Card bodyStyle={{padding: 0}}>
            <Tabs centered defaultActiveKey="ai">
              <Tabs.TabPane tab={<span className="tab-indicator"><ExperimentOutlined/><span>AI</span></span>} key="ai">
                <AIProperties/>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><BorderOuterOutlined/> <span>Depth</span></span>} key="depth">
                I'm Depth
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><CameraOutlined/> <span>Camera</span></span>} key="camera">
                I'm Camera
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><SlidersOutlined/> <span>Misc</span></span>} key="misc">
                I'm Misc
              </Tabs.TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default App;
