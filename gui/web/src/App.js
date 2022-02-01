import { Card, Col, Row, Tabs} from "antd";
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
import {fetchConfig} from "./store";
import {useEffect} from "react";
import CameraProperties from "./components/CameraProperties";
import MiscProperties from "./components/MiscProperties";

function App() {
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
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><BorderOuterOutlined/> <span>Depth</span></span>}
                            key="depth">
                <DepthProperties/>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><CameraOutlined/> <span>Camera</span></span>}
                            key="camera">
                <CameraProperties/>
              </Tabs.TabPane>
              <Tabs.TabPane tab={<span className="tab-indicator"><SlidersOutlined/> <span>Misc</span></span>}
                            key="misc">
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
