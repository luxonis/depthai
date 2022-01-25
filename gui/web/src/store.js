import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET} from "./request";

export const fetchConfig = createAsyncThunk(
  'config/fetch',
  async () => {
    const response = await request(GET, '/config')
    return response.data
  }
)

console.log(fetchConfig)

export const demoSlice = createSlice({
  name: 'demo',
  initialState: {
    fetched: false,
    config: {},
    error: null,
  },
  reducers: {
  },
  extraReducers: (builder) => {
    builder.addCase(fetchConfig.fulfilled, (state, action) => {
      state.config = action.payload
      state.fetched = true
    })
    builder.addCase(fetchConfig.rejected, (state, action) => {
      state.error = action.error
      state.fetched = true
    })
  },
})


export default configureStore({
  reducer: {
    demo: demoSlice.reducer,
  }
})