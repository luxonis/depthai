import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET, POST} from "./request";
import _ from 'lodash';

export const fetchConfig = createAsyncThunk(
  'config/fetch',
  async () => {
    const response = await request(GET, '/config')
    return response.data
  }
)

export const sendConfig = createAsyncThunk(
  'config/send',
  async (arg, thunk) => {
    const updates = thunk.getState().demo.updates
    console.log(updates)
    const response = await request(POST, '/update', updates)
    console.log("RESP", response.data)
  }
)

export const demoSlice = createSlice({
  name: 'demo',
  initialState: {
    fetched: false,
    restartRequired: false,
    config: {},
    updates: {},
    rawConfig: {},
    error: null,
  },
  reducers: {
    updateAIConfig: (state, action) => {
      state.config.ai = _.merge(state.config.ai || {}, action.payload)
      state.updates.ai = _.merge(state.updates.ai || {}, action.payload)
      state.restartRequired = true
    }
  },
  extraReducers: (builder) => {
    builder.addCase(sendConfig.pending, (state, action) => {
      state.fetched = false
      state.restartRequired = false
    })
    builder.addCase(sendConfig.fulfilled, (state, action) => {
      state.fetched = true
    })
    builder.addCase(fetchConfig.pending, (state, action) => {
      state.fetched = false
    })
    builder.addCase(fetchConfig.fulfilled, (state, action) => {
      state.config = action.payload
      state.rawConfig = action.payload
      state.fetched = true
    })
    builder.addCase(fetchConfig.rejected, (state, action) => {
      state.error = action.error
      state.fetched = true
    })
  },
})

export const { updateAIConfig } = demoSlice.actions;


export default configureStore({
  reducer: {
    demo: demoSlice.reducer,
  }
})