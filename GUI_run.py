import wx
from GUI_matching_distance import matching_distance

class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MyFrame, self).__init__(parent, title=title, size=(400, 200))

        panel = wx.Panel(self)

        self.btnLoadPlate = wx.Button(panel, label='Load Plate')
        self.btnLoadRadius = wx.Button(panel, label='Load Radius')
        self.btnStart = wx.Button(panel, label='Start')

        radius_choices = ["Left Radius", "Right Radius"]
        self.radius_choice = wx.Choice(panel, choices=radius_choices)
        self.radius_choice.SetSelection(0)

        plate_choices = ["Left Plate", "Right Plate"]
        self.plate_choice = wx.Choice(panel, choices=plate_choices)
        self.plate_choice.SetSelection(0)

        self.checkboxWatershed = wx.CheckBox(panel, label="Enable Watershedline Constraint")
        self.checkboxWatershed.SetValue(True)

        self.btnLoadPlate.Bind(wx.EVT_BUTTON, self.on_load_plate)
        self.btnLoadRadius.Bind(wx.EVT_BUTTON, self.on_load_radius)
        self.btnStart.Bind(wx.EVT_BUTTON, self.on_start)

        self.plate_file = None
        self.radius_file = None

        sizer = wx.BoxSizer(wx.VERTICAL)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self.btnLoadPlate, 0, wx.ALL, 5)
        button_sizer.Add(self.btnLoadRadius, 0, wx.ALL, 5)
        sizer.Add(button_sizer, 0, wx.CENTER)
        sizer.Add(self.plate_choice, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.radius_choice, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.checkboxWatershed, 0, wx.ALL | wx.CENTER, 5)
        sizer.AddStretchSpacer()
        sizer.Add(self.btnStart, 0, wx.ALL | wx.ALIGN_RIGHT, 5)
        panel.SetSizer(sizer)

        self.Centre()
        self.Show(True)

    def on_load_plate(self, event):
        with wx.FileDialog(self, "Open Plate STL File", wildcard="STL files (*.stl)|*.stl",
                        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            self.plate_file = fileDialog.GetPath()

    def on_load_radius(self, event):
        with wx.FileDialog(self, "Open Radius STL File", wildcard="STL files (*.stl)|*.stl",
                        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            self.radius_file = fileDialog.GetPath()
    
    def on_start(self, event):
        data = self.collect_data()
        print("Starting with data:", data)
        self.process_files(data)

    def collect_data(self):
        radius_selection = self.radius_choice.GetString(self.radius_choice.GetSelection())
        plate_selection = self.plate_choice.GetString(self.plate_choice.GetSelection())
        watershedline_constraint = self.checkboxWatershed.GetValue()
        data = {
            'plate_file': self.plate_file,
            'radius_file': self.radius_file,
            'radius_selection': radius_selection,
            'plate_selection': plate_selection,
            'watershedline_constraint': watershedline_constraint
        }
        return data

    def process_files(self, data):
        radius_stl = data['radius_file']
        plate_stl = data['plate_file']
        radius_selection = data['radius_selection']
        plate_selection = data['plate_selection']
        watershedline_constraint = data['watershedline_constraint']

        if radius_stl and plate_stl:
            matching_distance(radius_stl, plate_stl, radius_selection, plate_selection, watershedline_constraint)
        else:
            wx.MessageBox('Please load both STL files before starting.', 'Error', wx.OK | wx.ICON_ERROR)

class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, title="Plate Positioning")
        self.SetTopWindow(frame)
        return True

app = MyApp(False)
app.MainLoop()