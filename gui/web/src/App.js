import {Button, Card, Col, Input, Row, Select, Slider, Spin, Switch, Tabs, Typography} from "antd";
import {
  CameraOutlined,
  BorderOuterOutlined,
  ExperimentOutlined,
  PlayCircleOutlined,
  SlidersOutlined, QuestionCircleOutlined
} from "@ant-design/icons";
import AIProperties from "./components/AIProperties";
import DepthProperties from "./components/DepthProperties";
import CameraPreview from "./components/CameraPreview";
import {useDispatch, useSelector} from "react-redux";
import {fetchConfig} from "./store";
import {useEffect} from "react";
import CameraProperties from "./components/CameraProperties";
import MiscProperties from "./components/MiscProperties";

function App() {
  const fetched = useSelector((state) => state.demo.fetched)
  const error = useSelector((state) => state.demo.error)
  const dispatch = useDispatch()

  console.log(error)

  useEffect(() => {
    if(!fetched) {
      dispatch(fetchConfig())
    }
  }, [fetched])

  return (
    <div className="root-container">
      <Row align="middle" gutter={{md: 20, sm: 0}}>
        <Col md={12} sm={24}>
          {
            fetched || error
              ? <CameraPreview/>
              : <Spin tip="Loading..."/>
          }
        </Col>
        <Col md={12} sm={24}>
          <Card bodyStyle={{padding: 0}}>
            <Tabs animated centered defaultActiveKey="ai">
              <Tabs.TabPane tab={<span className="tab-indicator"><ExperimentOutlined/><span>AI</span></span>} key="ai">
                <AIProperties/>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><BorderOuterOutlined/> <span>Depth</span></span>} key="depth">
                <DepthProperties/>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><CameraOutlined/> <span>Camera</span></span>} key="camera">
                <CameraProperties/>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><SlidersOutlined/> <span>Misc</span></span>} key="misc">
                <MiscProperties/>
              </Tabs.TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default App;
