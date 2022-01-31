import React, {useEffect, useState} from 'react'
import {Button, Select, Spin} from "antd";
import {PlayCircleOutlined} from "@ant-design/icons";
import {useSelector} from "react-redux";

const emptyImg = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAADvCAYAAADcvIJsAAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw1AUhU9TpSIVBwsWcchQnSyIijhqFYpQIdQKrTqYvPQPmjQkKS6OgmvBwZ/FqoOLs64OroIg+APi6OSk6CIl3pcUWsR44fE+zrvn8N59gNCoMM3qGgc03TbTyYSYza2KoVcEEMUgQoDMLGNOklLwra976qa6i/Ms/74/q0/NWwwIiMSzzDBt4g3i6U3b4LxPHGElWSU+Jx4z6YLEj1xXPH7jXHRZ4JkRM5OeJ44Qi8UOVjqYlUyNeIo4pmo65QtZj1XOW5y1So217slfGM7rK8tcpzWMJBaxBAkiFNRQRgU24rTrpFhI03nCxz/k+iVyKeQqg5FjAVVokF0/+B/8nq1VmJzwksIJoPvFcT5GgNAu0Kw7zvex4zRPgOAzcKW3/dUGMPNJer2txY6A/m3g4rqtKXvA5Q4QfTJkU3alIC2hUADez+ibcsDALdC75s2tdY7TByBDs0rdAAeHwGiRstd93t3TObd/e1rz+wH4+3J2eba+jwAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAAuIwAALiMBeKU/dgAAAAd0SU1FB+YBGg8tJvd2VFsAAAAZdEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIEdJTVBXgQ4XAAABiklEQVR42u3BgQAAAADDoPlTn+AGVQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAxwDW+gABpsz3nwAAAABJRU5ErkJggg=="

const CameraPreview = () => {
  const fetched = useSelector((state) => state.demo.fetched)

  return (
    <div className="preview-container">
      {
        !fetched && <Spin tip="Loading..."/>
      }
      <img alt="camera-gui" className="stream-preview " src={fetched ? "/stream" : emptyImg}/>
      <div className="preview-controls">
        <Select defaultValue="depth">
          <Select.Option value="depth">Depth</Select.Option>
          <Select.Option value="color">Color</Select.Option>
          <Select.Option value="disparity" disabled>
            Disparity
          </Select.Option>
          <Select.Option value="left">left</Select.Option>
        </Select>
        <Select defaultValue="184212512512CF">
          <Select.Option value="184212512512CF">184212512512CF</Select.Option>
        </Select>
        <Button danger type="primary" shape="round" icon={<PlayCircleOutlined/>} size="large">
          Reload
        </Button>
      </div>
    </div>
  );
};

export default CameraPreview;