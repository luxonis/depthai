import {Button, Card, Col, Row, Tabs} from "antd";
import {
  CameraOutlined,
  BorderOuterOutlined,
  ExperimentOutlined,
  SlidersOutlined,
} from "@ant-design/icons";
import AIProperties from "./components/AIProperties";
import DepthProperties from "./components/DepthProperties";
import CameraPreview from "./components/CameraPreview";
import {useDispatch, useSelector} from "react-redux";
import {fetchConfig, sendConfig} from "./store";
import {useEffect} from "react";
import CameraProperties from "./components/CameraProperties";
import MiscProperties from "./components/MiscProperties";

function App() {
  const restartRequired = useSelector((state) => state.demo.restartRequired)
  const dispatch = useDispatch()

  useEffect(() => {
    dispatch(fetchConfig())
  }, [])

  return (
    <div className="root-container">
      <Row justify="space-around" align="top" gutter={{ xs: 8, sm: 16, md: 24, lg: 32 }}>
        <Col>
          <CameraPreview/>
        </Col>
        <Col className="rightColumn">
          <Card hoverable={false} bordered={false} bodyStyle={{padding: 0, maxWidth: 400}}>
            <Tabs className="tab-group" animated defaultActiveKey="ai">
              <Tabs.TabPane tab={<span className="tab-indicator"><ExperimentOutlined/><span>AI</span></span>} key="ai">
                <AIProperties/>
                <Button disabled={!restartRequired} onClick={() => dispatch(sendConfig())} className="restart-button" type="primary" block size="large">
                  Apply and Restart
                </Button>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><BorderOuterOutlined/> <span>Depth</span></span>}
                            key="depth">
                <DepthProperties/>
                <Button disabled={!restartRequired} onClick={() => dispatch(sendConfig())} className="restart-button" type="primary" block size="large">
                  Apply and Restart
                </Button>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><CameraOutlined/> <span>Camera</span></span>}
                            key="camera">
                <CameraProperties/>
                <Button disabled={!restartRequired} onClick={() => dispatch(sendConfig())} className="restart-button" type="primary" block size="large">
                  Apply and Restart
                </Button>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><SlidersOutlined/> <span>Misc</span></span>}
                            key="misc">
                <MiscProperties/>
                <Button disabled={!restartRequired} onClick={() => dispatch(sendConfig())} className="restart-button" type="primary" block size="large">
                  Apply and Restart
                </Button>
              </Tabs.TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default App;
