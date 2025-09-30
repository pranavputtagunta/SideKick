import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

import OnboardingScreen from '../screens/OnboardingScreen';
import QuestionsScreen from '../screens/QuestionsScreen';
// import SkillDetailScreen from '../screens/SkillDetailScreen';
// import RecordingScreen from '../screens/RecordingScreen';
// import ResultsScreen from '../screens/ResultsScreen';

const Stack = createStackNavigator();

export default function AppNavigator() {
    return (
        <NavigationContainer>
            <Stack.Navigator initialRouteName="Onboarding">
                <Stack.Screen name="Onboarding" component={OnboardingScreen} />
                <Stack.Screen name="Questions" component={QuestionsScreen} />
                {/* <Stack.Screen name="SkillDetail" component={SkillDetailScreen} />
                <Stack.Screen name="Recording" component={RecordingScreen} />
                <Stack.Screen name="Results" component={ResultsScreen} /> */}
            </Stack.Navigator>
        </NavigationContainer>
    );
}