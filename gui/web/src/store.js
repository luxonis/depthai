import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET} from "./request";

export const fetchConfig = createAsyncThunk(
  'config/fetch',
  async () => {
    const response = await request(GET, '/config')
    return response.data
  }
)

export const demoSlice = createSlice({
  name: 'demo',
  initialState: {
    fetched: false,
    config: {},
  },
  reducers: {
  },
  extraReducers: (builder) => {
    builder.addCase(fetchConfig.fulfilled, (state, action) => {
      state.config = action.payload
      state.fetched = true
    })
  },
})


export default configureStore({
  reducer: {
    demo: demoSlice.reducer,
  }
})