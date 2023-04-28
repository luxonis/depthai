import PySimpleGUI as sg
import threading

def ir_handler(dev):
    # Enable Register (0x01)
    strobe_opts = {'level': 0, 'edge': 1}
    strobe_type = 0  # default 0
    mode_opts = {'standby': 0, 'irdrive': 1, 'torch': 2, 'flash': 3}
    mode = 1     # default 0
    enable = {}
    enable['Led1'] = 0  # default 0
    enable['Led2'] = 0  # default 0
    enable['TxPin'] = 0  # default 1
    enable['Strobe'] = 1 # default 0
    enable['TorchTempPin'] = 0  # default 0

    # IVFM Register (0x02)
    uvlo_en = 0  # default 0
    ivfm_level = 0  # 0..7 = 2.9..3.6V, default 0
    ivfm_hyst_en = 0  # 1 = 50mV, default 0
    ivfm_mode = 1  # 0-disabled, 1-stop_and_hold, 2-down_mode, 3-up_and_down, default 1

    # LED1/LED2 Flash/Torch Brightness Register (0x03,0x04,0x05,0x06)
    # Note: LED2 override (set together with LED1) not enabled
    # Set raw values (0..127)
    brightness = {}
    brightness['FlashLed1'] = 0
    brightness['FlashLed2'] = 0
    brightness['TorchLed1'] = 0
    brightness['TorchLed2'] = 0

    # Boost Configuration Register (0x07)
    led_short_fault_detect_en = 1  # default 1
    boost_mode_opts = {'normal': 0, 'pass': 1}
    boost_mode = 0  # default 0
    boost_freq_opts = {'2MHz': 0, '4MHz': 1}
    boost_freq = 0  # default 0
    boost_limit_opts = {'1.9A': 0, '2.8A': 1}
    boost_limit = 1  # default 1

    # Timing Configuration Register (0x08)
    torch_ramp_time = 1  # 0-no_ramp, 1=1ms, 2=32ms, 3=64ms,... 7=1024ms, default 1
    # Note a flash timeout event requires reconfig - not handled yet
    flash_timeout = 10  # 0..9 = 10..100ms, 10..15 = 150..400ms, default 10

    # TEMP Register (0x09)
    torch_polarity = 0  # 0-high, 1-low, default 0
    ntc_open_fault_en = 0  # default 0
    ntc_short_fault_en = 0  # default 0
    temp_volt_thresh = 4  # 0..7 = 0.2..0.9V, default 4 = 0.6V
    torch_temp_sel = 0  # 0-torch, 1-temp, default 0

    def is_flash_en(): return mode in (1, 3)
    def is_torch_en(): return mode == 2
    def brightness_text(modeLed):
        v = brightness[modeLed]
        if modeLed.startswith('Flash'):
            return '{:8.3f} mA'.format(v * 11.725 + 10.9) \
                   + ('' if is_flash_en() else ' - Disabled')
        else:
            return '{:7.3f} mA'.format(v * 1.4 + 0.977) \
                   + ('' if is_torch_en() else ' - Disabled')
    def flash_timeout_text():
        return str((10*(flash_timeout+1)) if (flash_timeout<10) else (50*(flash_timeout-7))) + ' ms'

    def set_enable():
        v = (enable['TxPin'] << 7 | strobe_type << 6 | enable['Strobe'] << 5
            | enable['TorchTempPin'] << 4 | mode << 2 | enable['Led2'] << 1 | enable['Led1'])
        dev.irWriteReg(0x01, v)
    def set_ivfm():
        v = uvlo_en << 6 | ivfm_level << 3 | ivfm_hyst_en << 2 | ivfm_mode
        dev.irWriteReg(0x02, v)
    def set_brightness(modeLed):
        regs = {
            'FlashLed1': 0x03,
            'FlashLed2': 0x04,
            'TorchLed1': 0x05,
            'TorchLed2': 0x06,
        }
        dev.irWriteReg(regs[modeLed], brightness[modeLed])
    def set_boost():
        v = led_short_fault_detect_en << 3 | boost_mode << 2 | boost_freq << 1 | boost_limit
        dev.irWriteReg(0x07, v)
    def set_timing():
        v = torch_ramp_time << 4 | flash_timeout
        dev.irWriteReg(0x08, v)
    def set_temp():
        v = (torch_polarity << 6 | ntc_open_fault_en << 5 | ntc_short_fault_en << 4
            | temp_volt_thresh << 1 | torch_temp_sel)
        dev.irWriteReg(0x09, v)

    set_ivfm()
    set_brightness('FlashLed1')
    set_brightness('FlashLed2')
    set_brightness('TorchLed1')
    set_brightness('TorchLed2')
    set_boost()
    set_timing()
    set_temp()
    set_enable()

    sg.theme('Default1')

    layoutEnable = [
        [sg.Frame('Mode', [
            [sg.Radio('Standby',  'Rmode', key='standby', default=(mode==0), enable_events=True)],
            [sg.Radio('IR Drive', 'Rmode', key='irdrive', default=(mode==1), enable_events=True)],
            [sg.Radio('Torch',    'Rmode', key='torch',   default=(mode==2), enable_events=True)],
            [sg.Radio('Flash',    'Rmode', key='flash',   default=(mode==3), enable_events=True)],
            ]),
         sg.Column([
             [sg.Checkbox('TX Pin Enable', key='TxPin', default=enable['TxPin'], enable_events=True)],
             [sg.Checkbox('TORCH/TEMP Pin Enable', key='TorchTempPin', default=enable['TorchTempPin'], enable_events=True)],
             [sg.Checkbox('STROBE Enable:', key='Strobe', default=enable['Strobe'], enable_events=True),
              sg.Radio('Level', 'Rstr', key='level', default=(strobe_type==0), enable_events=True),
              sg.Radio('Edge',  'Rstr', key='edge',  default=(strobe_type==1), enable_events=True),
             ],
         ]),
        ],
    ]

    layoutBoostTiming = [
        [sg.Frame('Boost', [
            [sg.Text('Mode:'),
             sg.Radio('Normal',    'RBm', key='normal', default=(boost_mode==0), enable_events=True),
             sg.Radio('Pass only', 'RBm', key='pass',   default=(boost_mode==1), enable_events=True)],
            [sg.Text('Frequency:'),
             sg.Radio('2 MHz', 'RBfreq', key='2MHz', default=(boost_freq==0), enable_events=True),
             sg.Radio('4 MHz', 'RBfreq', key='4MHz', default=(boost_freq==1), enable_events=True)],
            [sg.Text('Current limit:'),
             sg.Radio('1.9 A', 'RBlimit', key='1.9A', default=(boost_limit==0), enable_events=True),
             sg.Radio('2.8 A', 'RBlimit', key='2.8A', default=(boost_limit==1), enable_events=True)],
            ]),
         sg.Frame('Flash timeout duration', [
            [sg.Slider(range=(0, 15), default_value=flash_timeout, orientation='h',
                       enable_events=True, key='FlashTimeout')],
            [sg.Text(flash_timeout_text(), key='TextFlashTimeout')],
            ]),
        ],
    ]

    layoutLed1 = [
        [sg.Checkbox('Enable', default=enable['Led1'], enable_events=True, key='Led1')],
        [sg.Text('Flash:'),
         sg.Slider(range=(0, 127), default_value=brightness['FlashLed1'], orientation='h',
                   size=(25, 18), enable_events=True, key='FlashLed1'),
         sg.Text(brightness_text('FlashLed1'), key='TextFlashLed1')
        ],
        [sg.Text('Torch:'),
         sg.Slider(range=(0, 127), default_value=brightness['TorchLed1'], orientation='h',
                   size=(25, 18), enable_events=True, key='TorchLed1'),
         sg.Text(brightness_text('TorchLed1'), key='TextTorchLed1')
        ],
    ]

    layoutLed2 = [
        [sg.Checkbox('Enable', default=enable['Led2'], enable_events=True, key='Led2')],
        [sg.Text('Flash:'),
         sg.Slider(range=(0, 127), default_value=brightness['FlashLed2'], orientation='h',
                   size=(25, 18), enable_events=True, key='FlashLed2'),
         sg.Text(brightness_text('FlashLed2'), key='TextFlashLed2')
        ],
        [sg.Text('Torch:'),
         sg.Slider(range=(0, 127), default_value=brightness['TorchLed2'], orientation='h',
                   size=(25, 18), enable_events=True, key='TorchLed2'),
         sg.Text(brightness_text('TorchLed2'), key='TextTorchLed2')
        ],
    ]

    layoutRegCtrl = [
        [sg.Text('Reg:'),   sg.InputText('0x', size=(10,1), key='InRegW'),
         sg.Text('Value:'), sg.InputText('0x', size=(10,1), key='InValW'),
         sg.Button('Write')],
        [sg.Text('Reg:'),   sg.InputText('0x', size=(10,1), key='InRegR'),
         sg.Button('Read'), sg.Text('...', key='TextValR')],
    ]

    layoutMain = [
        layoutEnable,
        layoutBoostTiming,
        [sg.Frame('LED1 / Flood IR', layoutLed1)],
        [sg.Frame('LED2 / Laser Pattern Projector', layoutLed2)],
        [sg.Frame('LM3644 Register Control', layoutRegCtrl)],
    ]

    window = sg.Window('IR Control', layoutMain)

    while True:
        event, values = window.read()
        #print('==EV', event, values)
        if event == sg.WIN_CLOSED:
            break
        if event == 'Write':
            reg = int(values['InRegW'], 0)
            val = int(values['InValW'], 0)
            dev.irWriteReg(reg, val)
        if event == 'Read':
            reg = int(values['InRegR'], 0)
            val = hex(dev.irReadReg(reg))
            window['TextValR'].update(val)
        if event in ('FlashLed1', 'FlashLed2', 'TorchLed1', 'TorchLed2'):
            brightness[event] = int(values[event])
            window['Text' + event].update(brightness_text(event))
            set_brightness(event)
        if event == 'FlashTimeout':
            flash_timeout = int(values['FlashTimeout'])
            window['TextFlashTimeout'].update(flash_timeout_text())
            set_timing()
        if event in ('Led1', 'Led2', 'Strobe', 'TxPin', 'TorchTempPin'):
            enable[event] = values[event]
            set_enable()
        if event in ('standby', 'irdrive', 'torch', 'flash'):
            mode = mode_opts[event]
            set_enable()
            for e in ('FlashLed1', 'FlashLed2', 'TorchLed1', 'TorchLed2'):
                window['Text' + e].update(brightness_text(e))
        if event in ('level', 'edge'):  strobe_type = strobe_opts[event]; set_enable()
        if event in ('normal', 'pass'): boost_mode  = boost_mode_opts[event];  set_boost()
        if event in ('2MHz', '4MHz'):   boost_freq  = boost_freq_opts[event];  set_boost()
        if event in ('1.9A', '2.8A'):   boost_limit = boost_limit_opts[event]; set_boost()

    window.close()
    exit()

if 0: # Change to 1 to enable stub GUI test (no device connection)
    class IrStub():
        def irWriteReg(self, reg, value):
            print("IR STUB write:", hex(reg), hex(value))
        def irReadReg(self, reg):
            print("IR STUB read :", hex(reg))
            return 0
    ir_handler(IrStub())

def start_ir_handler(dev):
    class IrLogWrapper:
        def __init__(self, dev):
            self.dev = dev
            self.idx = 1
        def irWriteReg(self, reg, value):
            print(f'===== Op{self.idx:3}: dev.irWriteReg({hex(reg)}, {hex(value)})')
            self.dev.irWriteReg(reg, value)
            self.idx += 1
        def irReadReg(self, reg):
            value = self.dev.irReadReg(reg)
            print(f'===== Op{self.idx:3}: dev.irReadReg({hex(reg)}) -> {hex(value)}')
            self.idx += 1
            return value

    t = threading.Thread(target=ir_handler, args=(IrLogWrapper(dev),))
    t.start()
