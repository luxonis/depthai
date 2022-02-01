import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET, POST} from "./request";
import _ from 'lodash';

export const fetchConfig = createAsyncThunk(
  'config/fetch',
  async (arg, thunk) => {
    const response = await request(GET, '/config')
    if (_.isEmpty(response.data)) {
      setTimeout(() => thunk.dispatch(fetchConfig()), 1000)
      return thunk.rejectWithValue('Empty');
    }
    return response.data
  }
)

function difference(object, base) {
    return _.transform(object, (result, value, key) => {
        if (!_.isEqual(value, base[key])) {
            result[key] = (_.isObject(value) && _.isObject(base[key])) ? difference(value, base[key]) : value;
        }
    });
}

export const sendConfig = createAsyncThunk(
  'config/send',
  async (act = true, thunk) => {
    const rawUpdates = thunk.getState().demo.updates
    const rawConfig = thunk.getState().demo.rawConfig
    const updates = difference(rawUpdates, rawConfig)
    await request(POST, '/update', updates)
    thunk.dispatch(fetchConfig())
  }
)

export const updatePreview = createAsyncThunk(
  'preview/send',
  async (act, thunk) => {
    const preview = thunk.getState().demo.config.preview.current
    await request(POST, '/updatePreview', {preview})
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
    },
    updatePreviewConfig: (state, action) => {
      state.config.preview = _.merge(state.config.preview || {}, action.payload)
      state.updates.preview = _.merge(state.updates.preview || {}, action.payload)
    }
  },
  extraReducers: (builder) => {
    builder.addCase(sendConfig.pending, (state, action) => {
      state.fetched = false
      state.restartRequired = false
    })
    builder.addCase(fetchConfig.pending, (state, action) => {
      state.fetched = false
    })
    builder.addCase(updatePreview.pending, (state, action) => {
      state.fetched = false
    })
    builder.addCase(updatePreview.fulfilled, (state, action) => {
      state.fetched = true
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

export const { updateAIConfig, updatePreviewConfig } = demoSlice.actions;


export default configureStore({
  reducer: {
    demo: demoSlice.reducer,
  }
})