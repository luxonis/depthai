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
  async (act, thunk) => {
    const updates = thunk.getState().demo.updates
    await request(POST, `/update?restartRequired=true`, updates)
    thunk.dispatch(fetchConfig())
  }
)

async function dynUpdateFun(act, thunk) {
  thunk.dispatch(toggleFetched(false))
  const updates = thunk.getState().demo.updatesDynamic
  await request(POST, `/update?restartRequired=false`, updates)
  thunk.dispatch(toggleFetched(true))
}

const debouncedHandler = _.debounce(dynUpdateFun, 400);

export const sendDynamicConfig = createAsyncThunk(
  'config/send-dynamic',
  debouncedHandler
)

export const demoSlice = createSlice({
  name: 'demo',
  initialState: {
    fetched: false,
    restartRequired: false,
    config: {},
    updates: {},
    updatesDynamic: {},
    rawConfig: {},
    error: null,
  },
  reducers: {
    toggleFetched: (state, action) => {
      state.fetched = action.payload
      if(action.payload) {
        state.updatesDynamic = {}
      }
    },
    updateConfig: (state, action) => {
      state.config = _.merge(state.config, action.payload)
      state.updates = _.merge(state.updates, action.payload)
      state.restartRequired = true
    },
    updateDynamicConfig: (state, action) => {
      state.config = _.merge(state.config, action.payload)
      state.updatesDynamic = _.merge(state.updatesDynamic, action.payload)
    },
  },
  extraReducers: (builder) => {
    builder.addCase(sendConfig.pending, (state, action) => {
      state.fetched = false
      state.restartRequired = false
    })
    builder.addCase(fetchConfig.pending, (state, action) => {
      state.fetched = false
    })
    builder.addCase(fetchConfig.fulfilled, (state, action) => {
      state.config = action.payload
      state.rawConfig = action.payload
      state.fetched = true
      state.updates = {}
      state.updatesDynamic = {}
    })
    builder.addCase(fetchConfig.rejected, (state, action) => {
      state.error = action.error
      state.fetched = true
    })
  },
})

export const {updateConfig, updateDynamicConfig, toggleFetched} = demoSlice.actions;


export default configureStore({
  reducer: {
    demo: demoSlice.reducer,
  }
})