import torch
import torch.nn as nn
import torch.nn.functional as F

class Regression_model(nn.Module):
    
    def __init__(self, backbone, RF_regressor = None):
        super().__init__()
        self.backbone = backbone
        self.linear_1 = nn.Linear(1536,768)
        self.linear_2 = nn.Linear(768,384)
        self.linear_3 = nn.Linear(384,1)
        self.main_input_name= "pixel_values"
        self.RF_regressor = RF_regressor

    def forward(self, pixel_values_1=None, pixel_values_2=None,
                pixel_values_3=None, 
                input_ids=None, labels=None,
               output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
               encoder_outputs=None,**model_kwargs):

        hidden_states_1 = self.backbone(pixel_values_1).last_hidden_state 
        hidden_states_1 = hidden_states_1.view(hidden_states_1.size(0), -1)
        
        hidden_states_2 = self.backbone(pixel_values_2).last_hidden_state    
        hidden_states_2 = hidden_states_2.view(hidden_states_2.size(0), -1)
        
        hidden_states_3 = self.backbone(pixel_values_3).last_hidden_state    
        hidden_states_3 = hidden_states_3.view(hidden_states_3.size(0), -1)
        
        cat = torch.cat((hidden_states_1, hidden_states_2, hidden_states_3), 1)
        
        if self.RF_regressor is None:
            outputs = self.linear_3(self.linear_2(self.linear_1(cat)))
            outputs = outputs.view(outputs.size(0))
        else:
            outputs = self.RF_regressor.predict([cat[0].detach().numpy()] )

        if labels is not None and not return_dict :
            
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(outputs, labels)
            
            return {"loss":loss,"logits":outputs}
        else:
            concat = torch.flatten(cat) # only to be used for RandomForestRegressor 
            return {"logits" :outputs,"labels":labels, "concat":concat}