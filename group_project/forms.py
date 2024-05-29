from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class MusicForm(FlaskForm):
    song_name = StringField('Song_name',
                           validators=[DataRequired(), Length(min=2, max=20)])
    song_year = StringField('Year',
                           validators=[DataRequired(), Length(min=1, max=20)])
    submit = SubmitField('Sign Up')
